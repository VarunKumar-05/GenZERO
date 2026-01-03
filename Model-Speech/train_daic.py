import sys
import argparse
import os

# Add project root to path to allow importing src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from collections import Counter, defaultdict

import torch
from src.preprocessing.daic_loader import DAICLoader
from src.bdh_snn.network import SpikingNeuralNetwork
from src.analysis.detector import AnalysisLayer
from src.memory.storage import PersistentMemory

# Add forked BDH repo to path
# external is in the project root
BDH_PATH = os.path.join(project_root, "external", "bdh")
if BDH_PATH not in sys.path:
    sys.path.append(BDH_PATH)
import bdh  # type: ignore


def train_daic(root_dir: str, max_sessions: int = None, target_session_id: str = None):
    print(f"Loading DAIC-WOZ data from {root_dir}...")
    loader = DAICLoader(root_dir, target_dim=128, audio_enabled=True, audio_feature_dim=45)
    
    # Filter for specific session if requested
    if target_session_id:
        # loader.sessions contains full paths
        filtered = [s for s in loader.sessions if target_session_id in os.path.basename(s)]
        if not filtered:
            print(f"Session {target_session_id} not found in {root_dir}")
            return
        loader.sessions = filtered
        print(f"Targeting specific session: {target_session_id}")
    # Optimization: slice sessions before loading to avoid processing everything
    elif max_sessions:
        loader.sessions = loader.sessions[:max_sessions]
        
    sessions = loader.iter_sessions()

    print(f"Discovered {len(sessions)} sessions: {[s.session_id for s in sessions]}")

    input_dim = sessions[0].features.shape[1] if sessions else 128
    print(f"Using input dimension: {input_dim}")
    
    # Use hidden_dim=256 to match the SER model capacity
    model = SpikingNeuralNetwork(input_dim=input_dim, hidden_dim=256)
    
    # Load learned synapses from SER if available
    synapse_path = "learned_synapses.pt"
    if os.path.exists(synapse_path):
        print(f"Loading learned synapses from {synapse_path}...")
        try:
            saved_weights = torch.load(synapse_path)
            # Verify shape
            if saved_weights.shape == model.synapses.weights.shape:
                model.synapses.weights.data = saved_weights
                print("Successfully transferred SER synaptic weights.")
            else:
                print(f"Shape mismatch: Saved {saved_weights.shape} vs Model {model.synapses.weights.shape}. Skipping.")
        except Exception as e:
            print(f"Failed to load synapses: {e}")
    else:
        print("No pre-trained synapses found. Using initialized weights.")

    analyzer = AnalysisLayer()
    memory = PersistentMemory()

    # Initialize byte-level BDH LM for emphasized words (uses forked repo)
    bdh_config = bdh.BDHConfig()
    bdh_model = bdh.BDH(bdh_config)

    session_reports = []
    
    # Global Aggregators
    global_state_words = defaultdict(Counter)
    global_state_sentences = defaultdict(list)
    
    for session in sessions:
        print(f"\nProcessing session {session.session_id}")
        features = torch.tensor(session.features, dtype=torch.float32)
        
        # Sort transcript by start time for efficient lookup
        sorted_transcript = sorted(session.transcript, key=lambda x: x['start'])

        last_weights = None
        session_anomalies = set()
        anomaly_counter = Counter()
        # Aggregators for words associated with specific states
        state_words = defaultdict(list)
        
        probe_stats = defaultdict(lambda: {"activation": [], "strength": [], "active": 0, "seen": 0})
        hidden_norms = []
        for idx, row in enumerate(features):
            x = row.unsqueeze(0)
            out = model.forward(x)
            hidden = out["hidden_state"]
            last_weights = model.update_synapses(x, hidden)
            analysis = analyzer.analyze(out, last_weights)

            if analysis.get("anomalies"):
                # Determine derived words based on time
                # Frame rate assumed ~5Hz (downsampled by 20 from ~100Hz) -> 0.2s per frame
                current_time = idx * 0.2 
                active_words = []
                for entry in sorted_transcript:
                    if entry['start'] <= current_time <= entry['stop']:
                        active_words.append(entry['text'])
                    elif entry['start'] > current_time:
                        break # Optimization
                
                if active_words:
                    # Tokenize properly
                    all_tokens = []
                    for sentence in active_words:
                        # Split and clean basic punctuation
                        words = sentence.lower().split()
                        for w in words:
                             clean = w.strip(".,?!\"")
                             if clean:
                                 all_tokens.append(clean)

                    # Filter: allow short words if they are disfluency markers
                    disfluency_markers = {"um", "uh", "er", "ah", "like", "well", "so"}
                    
                    meaningful_words = []
                    for w in all_tokens:
                        if w in disfluency_markers or (len(w) > 3 and w not in ["that", "this", "have", "with", "what"]):
                            meaningful_words.append(w)
                    
                    for anomaly in analysis["anomalies"]:
                        state_words[anomaly].extend(meaningful_words)
                        if active_words:
                            # Store unique sentences
                            # Clean up duplicates later or just append
                            global_state_sentences[anomaly].extend(active_words)
                            global_state_words[anomaly].update(meaningful_words)
                
                anomaly_counter.update(analysis["anomalies"])
            if analysis.get("concept_probes"):
                for concept, vals in analysis["concept_probes"].items():
                    probe_stats[concept]["seen"] += 1
                    probe_stats[concept]["activation"].append(vals.get("activation_level", 0.0))
                    probe_stats[concept]["strength"].append(vals.get("synaptic_strength", 0.0))
                    if vals.get("active"):
                        probe_stats[concept]["active"] += 1

            hidden_norms.append(hidden.norm().item())

            if idx % 200 == 0:
                print(f"  Frame {idx}: hidden norm={hidden.norm().item():.4f}")

        # Track emphasized words for this session
        memory.track_words(session.emphasized_words)
        if last_weights is not None:
            memory.save_state({"session": session.session_id, "weights": last_weights})

        # Build session summary
        summary = {
            "session": session.session_id,
            "frames": len(features),
            "hidden_norm_mean": float(sum(hidden_norms) / max(1, len(hidden_norms))),
            "anomalies": dict(anomaly_counter),
            "probes": {},
        }
        for concept, vals in probe_stats.items():
            seen = max(1, vals["seen"])
            summary["probes"][concept] = {
                "active_frac": vals["active"] / seen,
                "activation_mean": float(sum(vals["activation"]) / seen),
                "strength_mean": float(sum(vals["strength"]) / seen),
            }
        
        # Summarize state words (Top 10)
        summary["state_indicators"] = {}
        for state, words in state_words.items():
            common = Counter(words).most_common(10)
            summary["state_indicators"][state] = common # Store (word, count) tuples
            
        session_reports.append(summary)

    print("\nGlobal State-Determining Indicators:")
    for state, counter in global_state_words.items():
        print(f"\nState: {state}")
        print(f"  Top Determining Words: {counter.most_common(10)}")
        
        # Sample sentences
        sentences = global_state_sentences[state]
        # Get unique sentences to avoid repetition if frame sampling overlaps same sentence
        unique_sentences = list(set(sentences))
        # Shuffle or just pick first few? Let's pick a few distinct ones
        print(f"  Sample Determining Sentences: {unique_sentences[:3]} ... [Total: {len(unique_sentences)}]")

    print("\nSession-level reports:")
    for rep in session_reports:
        print(f"- {rep['session']} | frames={rep['frames']} | hidden_norm_mean={rep['hidden_norm_mean']:.4f}")
        if rep["anomalies"]:
            print(f"  anomalies: {rep['anomalies']}")
        if rep["probes"]:
            for concept, vals in rep["probes"].items():
                print(
                    f"  {concept}: active_frac={vals['active_frac']:.2f}, "
                    f"activation_mean={vals['activation_mean']:.3f}, strength_mean={vals['strength_mean']:.3f}"
                )
        if rep.get("state_indicators"):
            print("  State Indicators (most frequent words during state):")
            for state, words in rep["state_indicators"].items():
                print(f"    {state}: {words}")

    # Use BDH LM to generate a short sample conditioned on top determining words
    # Derive prompt from the most frequent state-determining words
    all_determining_words = Counter()
    for counter in global_state_words.values():
        all_determining_words.update(counter)
    
    top_prompt_words = all_determining_words.most_common(5)
    
    if top_prompt_words:
        prompt_words = " ".join([w for w, _ in top_prompt_words])
        prompt_bytes = torch.tensor(bytearray(prompt_words, "utf-8"), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            gen = bdh_model.generate(prompt_bytes, max_new_tokens=64, top_k=3)
        decoded = bytes(gen.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
        try:
            print("\nBDH text generation sample (conditioned on emphasized words):")
            print(decoded.encode('ascii', 'replace').decode('ascii'))
        except Exception as e:
            print(f"Could not print generated text due to encoding: {e}")

    # Save detailed report to file
    with open("daic_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("DAIC Analysis Report\n")
        f.write("====================\n\n")
        
        f.write("Global State-Determining Indicators:\n")
        for state, counter in global_state_words.items():
            f.write(f"\nState: {state}\n")
            f.write(f"  Top Determining Words: {', '.join([f'{w}({c})' for w, c in counter.most_common(15)])}\n")
            
            sentences = list(set(global_state_sentences[state]))
            f.write(f"  Sample Determining Sentences ({len(sentences)} total):\n")
            for s in sentences[:5]: # Show top 5 unique sentences
                f.write(f"    - \"{s}\"\n")
        
        f.write("\n" + "="*20 + "\n")
        f.write("Session Reports:\n")
        for rep in session_reports:
            f.write(f"\n- Session {rep['session']}\n")
            f.write(f"  Frames: {rep['frames']}\n")
            f.write(f"  Hidden Norm Mean: {rep['hidden_norm_mean']:.4f}\n")
            if rep["anomalies"]:
                f.write(f"  Anomalies Detected: {rep['anomalies']}\n")
            
            if rep.get("state_indicators"):
                f.write("  State Indicators (most frequent words during state):\n")
                for state, common_list in rep["state_indicators"].items():
                    # Format as word:count
                    formatted = ", ".join([f"{w}({c})" for w, c in common_list])
                    f.write(f"    {state}: {formatted}\n")
                    
    print("\nAnalysis report saved to 'daic_analysis_report.txt'")
    
    # Save the learned stable synapses
    save_path = "learned_synapses.pt"
    torch.save(model.synapses.weights.data, save_path)
    print(f"Freshly learned synapses saved to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Run DAIC-WOZ Analysis")
    parser.add_argument("--root", type=str, default=os.path.join("src", "Dataser", "DAIC_WOZ_Data"), help="Path to DAIC_WOZ_Data root")
    parser.add_argument("--session", type=str, default=None, help="Specific session ID to run (e.g. 300_P)")
    parser.add_argument("--max_sessions", type=int, default=None, help="Max number of sessions to process")
    
    args = parser.parse_args()
    
    train_daic(args.root, max_sessions=args.max_sessions, target_session_id=args.session)
