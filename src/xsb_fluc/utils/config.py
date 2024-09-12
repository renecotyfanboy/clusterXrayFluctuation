import os


class Config:

    def __init__(self):

        self.IS_NOTEBOOK = False
        self.IS_TITAN = False
        self.IS_OLYMPE = False

        if os.environ.get('IS_NOTEBOOK', False):

            self.DATA_PATH = 'data'
            self.RESULTS_PATH = 'results'
            self.IS_NOTEBOOK = True

        if os.environ.get('IS_TITAN', False):

            self.DATA_PATH = '/data/xifu/home/sdupourque/xcop/data'
            self.RESULTS_PATH = '/data/xifu/home/sdupourque/xcop/results'
            self.XMM_PATH = '/data/xifu/home/pointeco/xmm/arfrmfrsp'
            self.IS_TITAN = True

        if os.environ.get('IS_OLYMPE', False):

            self.DATA_PATH = os.path.join('/tmpdir', os.environ['USER'], 'data')
            self.RESULTS_PATH = os.path.join('/tmpdir', os.environ['USER'], 'results')
            self.IS_OLYMPE = True
            
        if os.environ.get('IS_NUWA', False):

            self.DATA_PATH = os.path.join('/xifu', os.environ['USER'], 'data')
            self.RESULTS_PATH = os.path.join('/xifu', os.environ['USER'], 'results')
            self.IS_NUWA = True



config = Config()
