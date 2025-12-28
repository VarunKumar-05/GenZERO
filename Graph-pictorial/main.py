import time
from src.input.audio import AudioStream
from src.input.video import VideoStream
from src.input.text import TextInput
from src.preprocessing.alignment import TemporalAlignmentLayer
from src.bdh_snn.network import SpikingNeuralNetwork
from src.analysis.detector import AnalysisLayer
from src.memory.storage import PersistentMemory
from src.clinical.dashboard import ClinicalDashboard
from src.edge.deploy import EdgeDeploy

def main():
    print("Starting Synaptix System...")
    
    # Initialize Components
    audio = AudioStream()
    video = VideoStream()
    text = TextInput()
    
    preprocessor = TemporalAlignmentLayer()
    snn = SpikingNeuralNetwork()
    analyzer = AnalysisLayer()
    memory = PersistentMemory()
    dashboard = ClinicalDashboard()
    edge = EdgeDeploy()
    
    print("\n--- System Loop Start ---")
    
    # 1. Acquisition
    a_data = audio.get_data()
    v_data = video.get_data()
    t_data = text.get_data()
    
    # 2. Preprocessing
    aligned_data = preprocessor.process(a_data, v_data, t_data)
    
    # 3. SNN Processing
    snn_output = snn.forward(aligned_data)
    
    # 4. Memory Interaction
    memory.save_state(snn_output)
    baseline = memory.get_baseline()
    
    # 5. Analysis
    analysis_results = analyzer.analyze(snn_output)
    
    # 6. Clinical Output
    report = dashboard.generate_report(analysis_results)
    print(report)
    
    # Visualize
    dashboard.visualize(a_data, snn_output)
    
    # 7. Edge Sync
    edge.sync()
    
    print("--- System Loop End ---")

if __name__ == "__main__":
    main()
