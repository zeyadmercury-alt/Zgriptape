from griptape.tools import BaseTool

class CalcTool(BaseTool):
    def calc_circumference(self, a:  float) -> float:
        return 2.0*3.14*a