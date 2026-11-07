"""NN Builder helper: start / stop / describe one EC2 instance with boto3. Credentials come from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY), set by the app from the Keychain for this call only. Prints one JSON line.

usage: python aws_ec2.py describe|start|stop <region> <instance-id>

Spot instances: describe also gives "lifecycle" ("spot" / "on-demand"), "state_reason" (e.g. Server.SpotInstanceShutdown = AWS stopped it
to take the capacity back) and, when allowed (ec2:DescribeSpotInstanceRequests), the spot request: type (persistent / one-time),
interruption behaviour (stop / hibernate / terminate) and its status code (e.g. instance-stopped-no-capacity). An instance AWS no longer
knows (terminated long ago) is reported as state "terminated".
"""
import json
import sys
import time


def main(action, region, instance_id):
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    ec2 = boto3.client("ec2", region_name=region)

    def describe():
        try:
            i = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                return {"state": "terminated", "host": "", "type": "", "lifecycle": "", "state_reason": "not found", "shutdown_behavior": None}
            raise
        out = {"state": i["State"]["Name"], "host": i.get("PublicDnsName") or i.get("PublicIpAddress") or "",
               "type": i.get("InstanceType", ""), "lifecycle": i.get("InstanceLifecycle", "on-demand"),
               "state_reason": i.get("StateReason", {}).get("Code", ""), "shutdown_behavior": None}
        if i.get("SpotInstanceRequestId"):
            try:
                r = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[i["SpotInstanceRequestId"]])["SpotInstanceRequests"][0]
                out["spot"] = {"type": r.get("Type", ""), "interruption": r.get("InstanceInterruptionBehavior", "terminate"),
                               "state": r.get("State", ""), "status": r.get("Status", {}).get("Code", ""),
                               "message": r.get("Status", {}).get("Message", "")}
            except (BotoCoreError, ClientError) as e:
                out["spot"] = {"error": str(e)}
        return out

    try:
        if action == "describe":
            out = describe()
            try:
                attr = ec2.describe_instance_attribute(InstanceId=instance_id, Attribute="instanceInitiatedShutdownBehavior")
                out["shutdown_behavior"] = attr["InstanceInitiatedShutdownBehavior"]["Value"]
            except (BotoCoreError, ClientError):
                pass
        elif action == "start":
            d = describe()
            if d["state"] == "terminated":
                raise SystemExit("the instance no longer exists (terminated)")
            if d.get("lifecycle") == "spot" and d.get("state_reason", "").startswith("Server.Spot"):
                # an interrupted spot instance cannot be started by hand: AWS starts it when capacity returns
                print(json.dumps({"ok": False, "error": "spot-interrupted", **d}))
                return 2
            if d["state"] == "stopping":
                # (a g4dn with a large disk took about 6 minutes to stop, 2026-11-07: wait up to 10)
                ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 120})
            ec2.start_instances(InstanceIds=[instance_id])
            ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
            out = describe()
            for _ in range(24):
                if out["host"]:
                    break
                time.sleep(5)
                out = describe()
        elif action == "stop":
            ec2.stop_instances(InstanceIds=[instance_id])
            out = describe()
        else:
            raise SystemExit("unknown action")
        print(json.dumps({"ok": True, **out}))
    except (BotoCoreError, ClientError) as e:
        msg = str(e)
        if "InsufficientInstanceCapacity" in msg:
            msg = "insufficient-capacity: " + msg
        print(json.dumps({"ok": False, "error": msg}))
        return 1
    except SystemExit as e:
        print(json.dumps({"ok": False, "error": str(e.code)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
