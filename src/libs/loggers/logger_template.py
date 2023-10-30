class LoggerTemplate():
    def __init__(self, *arg, **kwargs):
        raise NotImplementedError
    
    def upadate_loss(self, phase, value, step):
        raise NotImplementedError
    
    def update_metric(self, phase, metric, value, step):
        raise NotImplementedError
    
    