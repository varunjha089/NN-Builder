"""NN Builder helper: start / stop / describe one EC2 instance with boto3. Credentials come from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY), set by the app from the Keychain for this call only. Prints one JSON line.

usage: python aws_ec2.py describe|start|stop <region> <instance-id>
"""
import json
import sys
import time


def main(action, region, instance_id):
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    ec2 = boto3.client("ec2", region_name=region)

    def describe():
        i = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
        return {"state": i["State"]["Name"], "host": i.get("PublicDnsName") or i.get("PublicIpAddress") or "",
                "type": i.get("InstanceType", ""), "shutdown_behavior": None}

    try:
        if action == "describe":
            out = describe()
            try:
                attr = ec2.describe_instance_attribute(InstanceId=instance_id, Attribute="instanceInitiatedShutdownBehavior")
                out["shutdown_behavior"] = attr["InstanceInitiatedShutdownBehavior"]["Value"]
            except (BotoCoreError, ClientError):
                pass
        elif action == "start":
            if describe()["state"] == "stopping":
                ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
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
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
