from nanovllm_omni.config import EngineConfig
from nanovllm_omni.engine.model_runner import ModelRunner
from nanovllm_omni.outputs import OmniOutput
from nanovllm_omni.request import OmniRequest
from nanovllm_omni.sched.step_scheduler import StepScheduler


class OmniEngine:
    """
    engine是协调sheduler、modelrunner完成generate的主逻辑,将所有框架下的所有部分窜起来
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        # The current teaching engine executes one step for one request at a time.
        self.scheduler = StepScheduler(max_num_running_reqs=1)
        self.runner = ModelRunner(config)

    def generate(self, requests: OmniRequest | list[OmniRequest]) -> list[OmniOutput]:

        if isinstance(requests, OmniRequest):
            requests = [requests] 

        for request in requests: # 加入scheduler并由scheduler进行调度
            self.scheduler.add_request(request)

        outputs: list[OmniOutput] = []

        # 循环处理仍未完成的请求
        while self.scheduler.has_requests(): 
            """
            还缺少executor和worker
            """
            sched_output = self.scheduler.schedule() # 请求调度

            if sched_output.is_empty:
                break

            runner_output = self.runner.execute_stepwise(sched_output) # modelrunner执行模型推理

            # 向上返回模型处理结果,更新并决定下一轮requests
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output) 
            for finished_req_id in finished_req_ids:
                self.scheduler.pop_request_state(finished_req_id)

            # 记录请求结果并返回
            if runner_output.result is not None:
                outputs.append(runner_output.result)

        return outputs
