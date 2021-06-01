from .teeth_dataset import TeethDataset


class FacesDataset(TeethDataset):
    def __init__(self, opt):

        self.h = 256
        self.w = 256
        self.ffhq_dir = 'faces_256'
        self.braces_dir = 'braces_faces_256'
        TeethDataset.__init__(self, opt)
