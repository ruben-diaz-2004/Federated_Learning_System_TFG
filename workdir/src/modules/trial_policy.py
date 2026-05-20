import numpy as np
import syft as sy

def prepare():
    #server = sy.orchestra.launch(name="my_server1", port="auto", dev_mode=True, reset=False)
    do_client = sy.login(url="localhost",port=3300, email="info@openmined.org", password="changethis")
    do_client.register(
        email="newuser1@openmined.org", name="John Doe", password="pw", password_verify="pw"
    )
    ds_client = sy.login(url="localhost",port=3300, email="newuser1@openmined.org", password="pw")

    dataset = sy.Dataset(
        name="Dataset name",
        description="**Placehoder Dataset description**",
        asset_list=[sy.Asset(
            name="asset_name",
            data=[1,2,3], # real data
            mock=[4,5,6], # mock data
        )],
    )

    do_client.upload_dataset(dataset)

from typing import Any

class RepeatedCallPolicy(sy.CustomOutputPolicy):
    n_calls: int = 0
    downloadable_output_args: list[str] = []
    state: dict[Any, Any] = {}

    def __init__(self, n_calls=1, downloadable_output_args: list[str] = None):
        self.downloadable_output_args = (
            downloadable_output_args if downloadable_output_args is not None else []
        )
        self.n_calls = n_calls
        self.state = {"counts": 0}

    def public_state(self):
        return self.state["counts"]

    def update_policy(self, context, outputs):
        self.state["counts"] += 1

    def apply_to_output(self, context, outputs, update_policy=True):
        if hasattr(outputs, "syft_action_data"):
            outputs = outputs.syft_action_data
            print(f"Outputs: {outputs}")
        else:
            print(f"Outputs is not syft action data")

        output_dict = {}
        if self.state["counts"] < self.n_calls:
            for output_arg in self.downloadable_output_args:
                output_dict[output_arg] = outputs[output_arg]
                output_dict[output_arg].append(self.state["counts"])
            if update_policy:
                self.update_policy(context, outputs)
        else:
            return None
        return output_dict

    def is_valid(self, context):
        return self.state["counts"]>=self.n_calls

if __name__ == '__main__':
    #prepare()
    policy = RepeatedCallPolicy(n_calls=1, downloadable_output_args=["y"])
    print(policy.init_kwargs)
    a_obj = sy.ActionObject.from_obj({"y": [1, 2, 3]})
    x = policy.apply_to_output(None, a_obj)
    #print(x["y"])
    # Remove pending requests
    do_client = sy.login(url="localhost",port=3300, email="info@openmined.org", password="changethis")
    print("Cleaning all pending requests")
    for request in do_client.requests:
        print(f"Treating request: {request}")
        request.deny(reason="Denying all pending requests")
    

    # Cretae request as a data scientists
    ds_client = sy.login(url="localhost",port=3300, email="newuser1@openmined.org", password="pw")
    @sy.syft_function(
        input_policy=sy.ExactMatch(x=ds_client.datasets[0].assets[0]),
        output_policy=RepeatedCallPolicy(n_calls=3, downloadable_output_args=["y"]),
    )
    def func15(x):
        import numpy as np
        y=np.sum(x)
        return {"y":y, "z":1}
        #return x
    # Undo all pending requests
    for request in ds_client.requests:
        print(request.history)

    #print(f"Code status {ds_client.code.status()}")

    #ds_client.code.request_code_execution(func14)
    #ds_client.code.submit(func15)
    # Login as data owner
    # Approve as the data owner
    do_client = sy.login(url="localhost",port=3300, email="info@openmined.org", password="changethis")

    for request in do_client.requests:
        request.approve()
        #request.deny('Simply denying')

    ds_client.requests
    for request in ds_client.requests:
        print(f"Request status {request.get_status()}")
        print(f"Request history len {len(request.history)}")

    request = ds_client.requests[0]
    for _ in range(1):
        #result = request.code.run(x=ds_client.datasets[0].assets[0])
        #ds_client.code.submit(func3)
        result=(ds_client.code.func15(x=ds_client.datasets[0].assets[0])).get()
        print(result)
