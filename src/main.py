import torch
from diffusers import StableDiffusionPipeline, ModelMixin
import wx


class NoCheck(ModelMixin):
    """Can be used in place of safety checker. Use responsibly and at your own risk."""

    def __init__(self):
        super().__init__()
        self.register_parameter(name='asdf',
                                param=torch.nn.Parameter(torch.randn(3)))

    def forward(self, images=None, **kwargs):
        return images, [False]


class DiffusionThing:

    def __init__(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", use_auth_token=True)
        pipe.safety_checker = NoCheck()
        pipe.enable_attention_slicing()
        self.pipe = pipe.to("mps")

    def run(self, prompt: str):
        image = self.pipe(prompt).images[0]
        image.save('output.png')


class MyWin(wx.Frame):

    def __init__(self, parent):
        super(MyWin, self).__init__(parent,
                                    title="Stable Diffusion",
                                    size=(200, 150))
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.txt = wx.TextCtrl(panel, -1, size=(140, -1))
        self.txt.SetValue('Landscape')
        vbox.Add(self.txt, 0, 0, 0)

        self.btn = wx.Button(panel, -1, "click Me")
        self.btn.Bind(wx.EVT_BUTTON, self.OnClicked)
        vbox.Add(self.btn, 0, 0, 0)

        panel.SetSizer(vbox)

        self.Centre()
        self.Show()
        self.Fit()

    def OnClicked(self, event):
        # btn = event.GetEventObject().GetLabel()
        DiffusionThing().run(self.txt.GetValue())


# Next, create an application object.
app = wx.App()
MyWin(None)
app.MainLoop()
