
from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory(r"C:\Users\u9133908\Documents\GitHub\do\..py\GenDocks-main\aa\redpanda")
model_trainer.trainModel(num_objects=10, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)
