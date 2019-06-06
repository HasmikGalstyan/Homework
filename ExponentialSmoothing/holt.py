class HoltES:
    def __init__(self, alpha=0.1, beta=0.1, d=1):
        self.alpha = alpha
        self.beta = beta
        self.d = d

    def smooth(self, data):
        smoothed_data = []
        l = data[0]
        b = 0
        for index, datum in enumerate(data):
            smoothed_data.append(l + b * self.d)
            l_new = self.alpha * datum + (1 - self.alpha) * (l + b)
            b = self.beta * (l_new - l) + (1 - self.beta) * b
            l = l_new
        return smoothed_data
