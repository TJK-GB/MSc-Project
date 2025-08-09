from models.resnet50_gru import ViolenceModel
# from models.swin import ViolenceModel
# from models.swin_gru import ViolenceModel
# from models.cue_net import ViolenceModel

model = ViolenceModel().to(device)

