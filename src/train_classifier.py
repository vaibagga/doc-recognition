model = models.resnet50(pretrained=True, progress=True)
for param in model.parameters():
   param.requires_grad = False
