from dft_image.dft import DFT
from .complex_filter import ComplexFilter

class ProcessingFactory(object):
    def __init__(self,
                 name,
                 proc_steps: ComplexFilter):
        self.name = name
        self.proc_steps = proc_steps
        self.dft = DFT()

    def process(self,
                img):

        img = self.dft.dft(img)
        img = self.proc_steps.processing(img)
        img = self.dft.idft(img)

        return img



