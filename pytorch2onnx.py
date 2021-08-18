import torch
from myNet import myNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state_path = r"C:\train_data\isDriver\modelKd_pruned2\0.870647_epoth_97_model.pth.tar"
checkpoint = torch.load(model_state_path) # pytorch模型加载
cfg = checkpoint["cfg"]
net = myNet(num_classes=2, cfg=cfg)
net.load_state_dict(checkpoint["state_dict"])
batch_size = 1  #批处理大小
input_shape = (3, 128, 128)   #输入数据,改成自己的输入shape
net.cuda()
# #set the model to inference mode
net.eval()

x = torch.randn(batch_size, *input_shape)   # 生成张量
x = x.to(device)
export_onnx_file = "test.onnx"  # 目的ONNX文件名
torch.onnx.export(net,x,export_onnx_file,export_params=True)