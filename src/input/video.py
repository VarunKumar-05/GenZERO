class VideoStream:
    def __init__(self):
        self.fps = 30
        print("Initialized Video Stream")

    def get_data(self):
        # Simulate video pose data
        return {"type": "video", "pose": {"x": 10, "y": 20}}
