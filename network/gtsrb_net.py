
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.drop = nn.Dropout2d()


        self.fc1 = nn.Linear(1000, 350)
        self.fc2 = nn.Linear(350, nclasses)

        # self.vit1 = VisionTransformer(model_name='face_vit_64_1', img_size=32, patch_size=8, in_chans=3, embed_dim=256,
        #                               depth=8, num_heads=1, num_classes=43, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #                               drop_rate=0.1, attn_drop_rate=0., pos_embed_interp=False, random_init=False,
        #                               align_corners=False)
        # self.vit2 = VisionTransformer(model_name='face_vit_64_2', img_size=32, patch_size=8, in_chans=3, embed_dim=512,
        #                               depth=8, num_heads=8, num_classes=43, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #                               drop_rate=0.1, attn_drop_rate=0., pos_embed_interp=False, random_init=False,
        #                               align_corners=False)
        # self.vit3 = VisionTransformer(model_name='face_vit_64_3', img_size=32, patch_size=8, in_chans=3, embed_dim=512,
        #                               depth=8, num_heads=8, num_classes=43, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #                               drop_rate=0.1, attn_drop_rate=0., pos_embed_interp=False, random_init=False,
        #                               align_corners=False)
        # self.final1 = Mlp(in_features=1000, out_features=nclasses)

    def forward(self, x):
        x = self.drop(self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2)))
        x = self.drop(self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2)))
        x = self.drop(self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2)))

        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)