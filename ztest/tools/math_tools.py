from griptape.tools import BaseTool
from griptape.utils.decorators import activity

class MTools(BaseTool):
    @activity(config={"description": "add two numbers"})
    def add_two(self, a: int, b: int) -> int:
        return a+b
    
    @activity(config={"description": "multilpy three numbers"})
    def mull_three(self, a: int, b: int, c: int) -> int:
        return a*b*c