

class AverageMeter(object):
    """Basic meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        """ reset meter
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ incremental meter
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
