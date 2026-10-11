class FatigueEngine:
    def __init__(self):
        self.score = 0

    def update(self, drowsy, yawn, distracted):
        # weighted scoring system
        if drowsy:
            self.score += 25
        if yawn:
            self.score += 15
        if distracted:
            self.score += 20

        # natural recovery
        self.score -= 5
        self.score = max(0, min(100, self.score))

        return self.score

    def level(self):
        if self.score < 30:
            return "SAFE"
        elif self.score < 60:
            return "WARNING"
        else:
            return "DANGER"