

class DivinaPipelineModel:
    def __init__(self):
        pass

    def __eq__(self, other):
        for s, o in zip(self.stages, other.stages):
            if not s.get_params() == o.get_params():
                return False
        return True
