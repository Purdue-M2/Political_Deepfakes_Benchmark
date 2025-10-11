import os
from sggan.detection import DetectionComponent
import json
import yaml

class MulticlassNet(nn.Module):
            def __init__(self, num_classes=10):
               super(MulticlassNet, self).__init__()
               self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
               self.fc = nn.Sequential(
                     nn.Linear(2048, 512),
                     nn.ReLU(),
                     nn.Dropout(0.5),
                     nn.Linear(512, 512),
                     nn.ReLU(),
                     nn.Dropout(0.5),
                     nn.Linear(512, 128),
                     nn.ReLU(),
                     nn.Dropout(0.5),
                     nn.Linear(128, num_classes)
                     )																							                        )
            def forward(self, x):
                  out = self.features(x)
                  out = out.reshape(out.size(0), -1)
                  out = self.fc(out)
                  return out



def process_MMA(FLDR,AG,id):
            json_path=FLDR +AG
            with open(json_path) as f :
                   mm_asset=yaml.safe_load(f)
                   #try: 
                   DetectionComponent.onMessage(DetectionComponent,mm_asset)
                   #except:
                   #    print('YYEXCEPTION IN IMAGE ID ' + str(id))

FLDR='/mnt/datalake/default/ag/'
list_dir = os.listdir(FLDR)
slist=sorted(list_dir)
print(len(slist))

DetectionComponent.onInit(DetectionComponent," ")

exit()
############# INIT NET
#if model_load_path and not os.path.exists(model_load_path):
            #raise Exception("Model path {} not found".format(model_load_path))

#if not model_load_path:
setup_type=1
is_binary=False
model_load_path = "models/setup"+str(setup_type)+("_binary.pth" if is_binary else "_multiclass.pth")
#if not os.path.exists(model_load_path):
#                   raise Exception("Model path {} not found".format(model_load_path))

resize_dims = 128

transform_img = transforms.Compose([
                    transforms.Resize(resize_dims),
                    transforms.CenterCrop(resize_dims),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ) ])
device = torch.device("cpu") if not use_cuda else torch.device("cuda:0")
checkpoint = torch.load(model_load_path, map_location=device)
model = nn.DataParallel(MulticlassNet(num_classes=num_classes)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

####################################
for i in range(len(slist)):
           if i>285:
                   print('PROCESSING ' + str(i) + ' ' +  slist[i])
                   process_MMA(FLDR,slist[i],i,model)

exit()

AG='faf8dcdde7744173f1eb54b26802d2f1ffd8ec3f-ag.json'
#AG='36bdee76d6d2f9ffbbd00b111e386d7edfee92ba-ag.json'
#AG='3d9edc00d9de8c9342cc145d4073a0974b6edee2-ag.json'
#AG='6e2f770432cd50907b677bb1c4af8cb64dfddbd5-ag.json'
#AG='7867dd3d6c61a1a0ac1700a78724bf74a21e52e0-ag.json'
#AG='79a0e8aa8b7b18c82ca4ab9d7aac63bf075655a1-ag.json'
#AG='b300bb079fc63f538065e24006d4f2c4075d3450-ag.json'

json_path='/mnt/datalake/default/ag/' +AG

### old way, the yaml is better
#mm_asset = json.load(open(json_path))
#nodes=mm_asset['nodes']
#Snodes = sorted(nodes, key=lambda x: x['nodeType'])
#print(Snodes)

with open(json_path) as f :
      mm_asset=yaml.safe_load(f)


DetectionComponent.onMessage(DetectionComponent,mm_asset)

