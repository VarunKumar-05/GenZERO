# Chrono-Synaptic Bio-Marker Dataflow

This diagram illustrates the data processing pipeline for the Chrono-Synaptic Bio-Marker system, utilizing the Pathway Dragon Hatchling (BDH) architecture.

```mermaid
graph TD
    %% Nodes and Subgraphs
    subgraph Input_Layer [Input Layer: Dual-Stream Ingestion]
        direction TB
        Patient((Patient))
        Audio[Audio Stream<br/>44kHz - Acoustic Physics]
        Text[Text/Video Stream<br/>30Hz - Semantic Logic]
        Pathway[Pathway Streaming Engine<br/>Synchronization]
        
        Patient --> Audio
        Patient --> Text
        Audio --> Pathway
        Text --> Pathway
    end

    subgraph Encoding_Layer [Encoding Layer: Spiking Neural Encoding]
        direction TB
        Sync[Synchronized Data Stream]
        SNN_Enc[Spiking Neural Encoder<br/>Pitch/Freq to Temporal Spikes]
        Spikes[Temporal Spike Train]
        
        Pathway --> Sync
        Sync --> SNN_Enc
        SNN_Enc --> Spikes
    end

    subgraph BDH_Core [BDH Architecture: Dual-Memory Topography]
        direction TB
        LIF[LIF Neurons<br/>Leaky Integrate-and-Fire]
        ISI_Detect[ISI Anomaly Detection<br/>Inter-Spike Interval]
        
        subgraph Memory_Systems [Dual-Memory]
            Frozen[Frozen Graph<br/>Clinical Gold Standard<br/>(Pre-trained)]
            Plastic[Plastic Graph<br/>User Baseline<br/>(Adaptive)]
        end
        
        Hebbian[Hebbian Learning<br/>Synaptic Weight Update]
        
        Spikes --> LIF
        LIF --> ISI_Detect
        ISI_Detect --> Frozen
        ISI_Detect --> Plastic
        Plastic -.->|Adaptation| Hebbian
        Hebbian -.->|Update Weights| Plastic
    end

    subgraph Analysis_Layer [Analysis: Differential Diagnosis]
        direction TB
        Comparator{Micro-Drift<br/>Detector}
        Drift_Signal[Drift Signal<br/>Deviation from Baseline]
        
        Frozen -->|Reference Signal| Comparator
        Plastic -->|User Signal| Comparator
        Comparator --> Drift_Signal
    end

    subgraph Output_Layer [Output: Clinical Insights]
        direction TB
        Alert_Decay[Alert: Semantic<br/>Coherence Decay]
        Alert_Jitter[Alert: Motor<br/>Control Jitter]
        Interp_Map[Sparse Interpretability Map<br/>'Glass Brain' Visualization]
        
        Drift_Signal --> Alert_Decay
        Drift_Signal --> Alert_Jitter
        Drift_Signal --> Interp_Map
    end
    
    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef memory fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef analysis fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px;

    class Patient,Audio,Text,Pathway input;
    class Sync,SNN_Enc,Spikes,LIF,ISI_Detect process;
    class Frozen,Plastic,Hebbian memory;
    class Comparator,Drift_Signal analysis;
    class Alert_Decay,Alert_Jitter,Interp_Map output;
```

## Key Components

1.  **Dual-Stream Ingestion**: Synchronizes high-frequency audio (acoustic physics) with lower-frequency video/text (semantic logic) using Pathway.
2.  **Spiking Neural Encoding**: Converts continuous signals into temporal spikes, preserving micro-timing information (e.g., tremors).
3.  **BDH Architecture**:
    *   **LIF Neurons**: Detect Inter-Spike Interval (ISI) anomalies.
    *   **Frozen Graph**: Prevents pathological overfitting by maintaining a clinical "gold standard".
    *   **Plastic Graph**: Adapts to the specific user via Hebbian learning, creating a "digital twin".
4.  **Differential Diagnosis**: Compares the user's current state against their established baseline to detect "Micro-Drift".
5.  **Outputs**: Generates specific alerts for neurological decay and provides a transparent "Glass Brain" visualization.
