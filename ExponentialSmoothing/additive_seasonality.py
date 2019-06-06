class AdditiveSeasonalityES:
    def __init__(self, alpha=0.1, beta=0.1, d=1, p=30):
        self.alpha = alpha
        self.beta = beta
        self.d = d
        self.p = p

    def smooth(self, data):
        smoothed_data = []
        s = []
        l = data[0]
        for index, datum in enumerate(data):
            if index - self.p + self.d % self.p<0:
                smoothed_data.append(l)
            else:
                smoothed_data.append(l + s[index - self.p + self.d % self.p])
            if index - self.p < 0:
                l_new = self.alpha * datum + (1 - self.alpha) * l
                s.append(self.beta * (datum - l))
            else:
                l_new = self.alpha * (datum - s[index - self.p]) + (1 - self.alpha) * l
                s.append(self.beta * (datum - l) + (1 - self.beta) * s[index - self.p])
            l = l_new
        return smoothed_data
