from .teeth_dataset import TeethDataset


class CroppedDataset(TeethDataset):
    def __init__(self, opt):

        self.h = int(128 * 1.5)
        self.w = int(256 * 1.5)
        self.ffhq_dir = 'cropped'
        self.braces_dir = 'braces_cropped'
        TeethDataset.__init__(self, opt)
