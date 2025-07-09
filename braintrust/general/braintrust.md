This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: py/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
py/
  src/
    braintrust/
      cli/
        install/
          __init__.py
          api.py
          logs.py
          redshift.py
        __main__.py
        eval.py
      __init__.py
      aws.py
      cache.py
      framework.py
      gitutil.py
      logger.py
      merge_row_batch.py
      oai.py
      resource_manager.py
      util.py
      version.py
  .gitignore
  README.md
  setup.py
```

# Files

## File: py/src/braintrust/cli/install/__init__.py
````python
import argparse
import textwrap

_module_not_found_error = None
try:
    from . import api, logs, redshift
except ModuleNotFoundError as e:
    _module_not_found_error = e


def fail_with_module_not_found_error(*args, **kwargs):
    raise ModuleNotFoundError(
        textwrap.dedent(
            f"""\
            At least one dependency not found: {str(_module_not_found_error)!r}
            It is possible that braintrust was installed without the CLI dependencies. Run:

              pip install 'braintrust[cli]'

            to install braintrust with the CLI dependencies (make sure to quote 'braintrust[cli]')."""
        )
    )


def build_parser(subparsers, parent_parser):
    install_parser = subparsers.add_parser(
        "install",
        help="Tools to setup and verify Braintrust's installation in your environment.",
        parents=[parent_parser],
    )
    if _module_not_found_error:
        install_parser.add_argument("args", nargs=argparse.REMAINDER)
        install_parser.set_defaults(func=fail_with_module_not_found_error)
    else:
        install_subparsers = install_parser.add_subparsers(dest="install_subcommand", required=True)

        for module in [api, logs, redshift]:
            module.build_parser(install_subparsers, parents=[parent_parser])
````

## File: py/src/braintrust/cli/install/api.py
````python
import logging
import textwrap
import time

from botocore.exceptions import ClientError

from ...aws import cloudformation

_logger = logging.getLogger("braintrust.install.api")

PARAMS = {
    "OrgName": "org_name",
    "ProvisionedConcurrency": "provisioned_concurrency",
    "DwDatabase": "dw_database",
    "DwHost": "dw_host",
    "DwUsername": "dw_username",
    "DwPassword": "dw_password",
    "DwPort": "dw_port",
    "DwType": "dw_type",
    "ManagedKafka": "managed_kafka",
    "KafkaBroker": "kafka_broker",
    "KafkaTopic": "kafka_topic",
    "KafkaUsername": "kafka_username",
    "KafkaPassword": "kafka_password",
}

DEFAULTS = {
    "ManagedKafka": "true",
    "DwType": "Postgres",
    "ProvisionedConcurrency": 0,
}

CAPABILITIES = ["CAPABILITY_IAM", "CAPABILITY_AUTO_EXPAND"]

LATEST_TEMPLATE = "https://braintrust-cf.s3.amazonaws.com/braintrust-latest.yaml"


def build_parser(subparsers, parents):
    parser = subparsers.add_parser("api", help="Install the Braintrust function API", parents=parents)

    parser.add_argument("name", help="Name of the CloudFormation stack to create or update")
    parser.add_argument(
        "--create",
        help="Create the stack if it does not exist",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--vpc-connect",
        help="Connect to an existing VPC",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--template",
        help="A specific CloudFormation template URL to use",
        default=None,
    )
    parser.add_argument(
        "--update-template",
        help="Update the CloudFormation to the latest version of the template",
        action="store_true",
        default=False,
    )

    # OrgName, ProvisionedConcurrency
    parser.add_argument("--org-name", help="The name of your organization", default=None)
    parser.add_argument(
        "--provisioned-concurrency",
        help="The amount of provisioned concurrency",
        default=None,
        type=int,
    )

    # DwHost, DwPort, DwPassword, DwPort, DwType, DwUsername
    parser.add_argument("--dw-database", help="The database of the data warehouse", default=None)
    parser.add_argument("--dw-host", help="The host of the data warehouse", default=None)
    parser.add_argument("--dw-username", help="The username of the data warehouse", default=None)
    parser.add_argument("--dw-password", help="The password of the data warehouse", default=None)
    parser.add_argument("--dw-port", help="The port of the data warehouse", default=None)
    parser.add_argument(
        "--dw-type",
        help="The type of the data warehouse",
        default=None,
        choices=[None, "Postgres", "Redshift", "Snowflake"],
    )

    # PostgresUrl
    parser.add_argument(
        "--managed-postgres",
        help="Spin up an RDS instance to use as the datastore",
        default=None,
        choices=[None, "true", "false"],
    )
    parser.add_argument(
        "--postgres-url", help="The postgres URL to use (if you are connecting to another VPC)", default=None
    )

    # ManagedKafka, KafkaBroker, KafkaTopic, KafkaUsername, KafkaPassword
    parser.add_argument(
        "--managed-kafka",
        help="Whether to include a managed Kafka (MSK)",
        default=None,
        choices=[None, "true", "false"],
    )
    parser.add_argument("--kafka-broker", help="The Kafka broker to use", default=None)
    parser.add_argument("--kafka-topic", help="The Kafka topic to use", default=None)
    parser.add_argument("--kafka-username", help="The Kafka username to use", default=None)
    parser.add_argument("--kafka-password", help="The Kafka password to use", default=None)

    # ElastiCacheClusterId
    parser.add_argument("--elasticache-cluster-host", help="The ElastiCacheCluster host to use", default=None)
    parser.add_argument(
        "--elasticache-cluster-port", help="The ElastiCacheCluster host to use", default=None, type=int
    )

    # SecurityGroupId, SubnetIds
    parser.add_argument("--security-group-id", help="The security group ID to use", default=None)
    parser.add_argument("--subnet-ids", help="The subnet IDs to use", default=None)

    parser.set_defaults(func=main)


def main(args):
    template = args.template or LATEST_TEMPLATE

    status = None
    try:
        statuses = cloudformation.describe_stacks(StackName=args.name)["Stacks"]
        if len(statuses) == 1:
            status = statuses[0]
        _logger.debug(status)
    except ClientError as e:
        if "does not exist" not in str(e):
            raise

    vpc_connect = args.vpc_connect
    if status and not vpc_connect:
        vpc_connect = "SecurityGroupId" in set(x["ParameterKey"] for x in status["Parameters"])

    if vpc_connect:
        PARAMS["SecurityGroupId"] = "security_group_id"
        PARAMS["SubnetIds"] = "subnet_ids"
        PARAMS["ElastiCacheClusterHost"] = "elasticache_cluster_host"
        PARAMS["ElastiCacheClusterPort"] = "elasticache_cluster_port"
        PARAMS["PostgresUrl"] = "postgres_url"

        if args.template is None:
            template = "https://braintrust-cf.s3.amazonaws.com/braintrust-latest-vpc.yaml"

    exists = status is not None
    if exists and args.create:
        _logger.error(
            textwrap.dedent(
                f"""\
            Stack with name {args.name} already exists. Either delete it in the AWS console or
            remove the --create flag."""
            )
        )
        exit(1)
    elif not exists and not args.create:
        _logger.error(
            textwrap.dedent(
                f"""\
            Stack with name {args.name} does not exist. Either create it manually by following
            https://www.braintrustdata.com/docs/getting-started/install or use the --create flag."""
            )
        )
        exit(1)

    if not exists:
        _logger.info(f"Creating stack with name {args.name}")
        cloudformation.create_stack(
            StackName=args.name,
            TemplateURL=template,
            Parameters=[
                {
                    "ParameterKey": param,
                    "ParameterValue": str(args.__dict__[arg_name] or DEFAULTS.get(param, "")),
                }
                for (param, arg_name) in PARAMS.items()
            ],
            Capabilities=CAPABILITIES,
        )

        for _ in range(120):
            status = cloudformation.describe_stacks(StackName=args.name)["Stacks"][0]
            if status["StackStatus"] != "CREATE_IN_PROGRESS":
                exists = True
                break
            _logger.info("Waiting for stack to be created...")
            time.sleep(5)
        else:
            _logger.error(
                textwrap.dedent(
                    """\
                Stack creation timed out. Please check the AWS console to see its status. You can also
                re-run this command without --create to continue the setup process once it's done."""
                )
            )
            exit(1)
        _logger.info(f"Stack with name {args.name} has been created with status: {status['StackStatus']}")
        exit(0)

    _logger.info(f"Stack with name {args.name} has status: {status['StackStatus']}")

    if not ("_COMPLETE" in status["StackStatus"] or "_FAILED" in status["StackStatus"]):
        _logger.info(f"Please re-run this command once the stack has finished creating or updating")
        exit(0)

    # Update params that have changed
    param_updates = {}
    for param, arg_name in PARAMS.items():
        if args.__dict__[arg_name] is not None:
            param_updates[param] = args.__dict__[arg_name]
    if len(param_updates) > 0 or args.update_template:
        template_kwargs = {"TemplateURL": template} if args.update_template else {"UsePreviousTemplate": True}
        _logger.info(
            f"Updating stack with name {args.name} with params: {param_updates} and template: {template_kwargs}"
        )
        cloudformation.update_stack(
            StackName=args.name,
            Parameters=[
                {"ParameterKey": param, "ParameterValue": str(update)} for (param, update) in param_updates.items()
            ]
            + [
                {"ParameterKey": param, "UsePreviousValue": True}
                for param in PARAMS.keys()
                if param not in param_updates
            ],
            Capabilities=CAPABILITIES,
            **template_kwargs,
        )

        for _ in range(120):
            status = cloudformation.describe_stacks(StackName=args.name)["Stacks"][0]
            if status["StackStatus"] != "UPDATE_IN_PROGRESS":
                exists = True
                break
            _logger.info("Waiting for stack to be updated...")
            time.sleep(5)
        else:
            _logger.error(
                textwrap.dedent(
                    """\
                Stack update timed out. Please check the AWS console to see its status. You can also
                re-run this command to try again."""
                )
            )
            exit(1)

        function_url = [x for x in status["Outputs"] if x["OutputKey"] == "EndpointURL"]
        if function_url:
            function_url = function_url[0]["OutputValue"]
        else:
            function_url = None
        _logger.info(f"Stack with name {args.name} has been updated with status: {status['StackStatus']}")
        _logger.info(f"Endpoint URL: {function_url}")
````

## File: py/src/braintrust/cli/install/logs.py
````python
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ...aws import cloudformation, logs

_logger = logging.getLogger("braintrust.install.logs")


def build_parser(subparsers, parents):
    parser = subparsers.add_parser("logs", help="Capture recent logs", parents=parents)
    parser.add_argument("name", help="Name of the CloudFormation stack to collect logs from")
    parser.add_argument("--service", help="Name of the service", default="api", choices=["api"])
    parser.add_argument("--hours", help="Number of seconds in the past to collect logs from", default=1, type=float)
    parser.set_defaults(func=main)


def main(args):
    stacks = cloudformation.describe_stacks(StackName=args.name)["Stacks"]
    if len(stacks) == 0:
        raise ValueError(f"Stack with name {args.name} does not exist")
    if len(stacks) > 1:
        raise ValueError(f"Multiple stacks with name {args.name} exist")
    stack = stacks[0]
    _logger.debug(stack)

    log_group_name = None
    if args.service == "api":
        lambda_function = [x for x in stack["Outputs"] if x["OutputKey"] == "APIHandlerName"]
        if len(lambda_function) != 1:
            raise ValueError(f"Expected 1 APIHandlerName, found {len(lambda_function)} ({lambda_function}))")
        log_group_name = f"/aws/lambda/{lambda_function[0]['OutputValue']}"
    else:
        raise ValueError(f"Unknown service {args.service}")

    start_time = int(time.time() - 3600 * args.hours) * 1000

    all_streams = []
    first_start_time = None
    nextToken = None

    while first_start_time is None or first_start_time >= start_time:
        kwargs = {}
        if nextToken is not None:
            kwargs["nextToken"] = nextToken

        stream_resp = logs.describe_log_streams(logGroupName=log_group_name, descending=True, **kwargs)

        first_start_time = min(s["firstEventTimestamp"] for s in stream_resp["logStreams"])
        nextToken = stream_resp.get("nextToken")

        streams = [s for s in stream_resp["logStreams"] if s["firstEventTimestamp"] >= start_time]
        streams.sort(key=lambda x: x["firstEventTimestamp"])
        all_streams = streams + all_streams

    _logger.debug(all_streams)

    def get_events(stream):
        return logs.get_log_events(
            logGroupName=log_group_name,
            logStreamName=stream["logStreamName"],
            startTime=start_time,
            startFromHead=True,
        )

    with ThreadPoolExecutor(8) as executor:
        events = executor.map(get_events, all_streams)

    last_ts = None
    for stream, log in zip(all_streams, events):
        print(f"---- {stream['logStreamName']}")
        for event in log["events"]:
            print(event)
````

## File: py/src/braintrust/cli/install/redshift.py
````python
import json
import logging
import re
import textwrap
from hashlib import md5

from ... import log_conn, login
from ...aws import iam, redshift_serverless

_logger = logging.getLogger("braintrust.install.redshift")


def build_parser(subparsers, parents):
    parser = subparsers.add_parser(
        "redshift",
        help="Setup Redshift to ingest from Braintrust (Kafka)",
        parents=parents,
    )

    parser.add_argument("name", help="Name of the Redshift cluster (or namespace) to create or update")
    parser.add_argument(
        "--create",
        help="Create the Redshift instance if it does not exist",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--serverless",
        help="Use Serverless Redshift",
        action="store_true",
        default=False,
    )
    parser.add_argument("--iam-role", help="IAM Role that can read from Kafka", default=None)
    parser.add_argument(
        "--iam-policy",
        help="Inline IAM policy permitting access to Kafka",
        default="BraintrustMSKReadPolicy",
    )
    parser.add_argument(
        "--msk-cluster-arn",
        help="The ARN of a specific MSK cluster to allow access to. If this flag is unspecified, Redshift can read from any MSK cluster in this AWS account",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--msk-topic-name",
        help="The name of a specific MSK topic to map into Redshift. The policy will allow access to all topics in the cluster, to support future topics",
        default="braintrust",
    )

    parser.add_argument(
        "--org-name",
        help="The name of your organization (optional, only needed if you belong to multiple orgs)",
    )

    parser.set_defaults(func=main)


def main(args):
    if args.create:
        raise NotImplementedError("Creating Redshift clusters is not yet supported")

    if args.msk_topic_name.lower() != args.msk_topic_name:
        raise ValueError("Kafka topic names must be lowercase (b/c of Redshift case sensitivity issues)")

    role_name = args.iam_role or ("bt-redshift-" + md5(args.msk_cluster_arn.encode("utf-8")).hexdigest())
    role = None
    try:
        role = iam.get_role(RoleName=role_name)
    except iam.exceptions.NoSuchEntityException:
        pass

    if role is None:
        _logger.info("Creating IAM Role %s", role_name)
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "redshift.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Braintrust Redshift Kafka Reader",
        )

    role_policy = None
    try:
        role_policy = iam.get_role_policy(RoleName=role_name, PolicyName=args.iam_policy)
    except iam.exceptions.NoSuchEntityException:
        pass

    # See definitions here: https://docs.aws.amazon.com/msk/latest/developerguide/iam-access-control.html
    msk_cluster_arn = args.msk_cluster_arn
    account_info, path = msk_cluster_arn.rsplit(":", 1)
    cluster_ident, cluster_name, cluster_uuid = path.split("/")
    if cluster_ident != "cluster":
        raise ValueError(f"Invalid MSK cluster ARN: {msk_cluster_arn}")

    # Allow access to all topics
    msk_topic_arn = f"{account_info}:topic/{cluster_name}/{cluster_uuid}/*"

    if role_policy is None:
        _logger.info(f"Creating inline IAM Policy {args.iam_policy} on {role_name}")

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "MSKIAMpolicy",
                    "Effect": "Allow",
                    "Action": [
                        "kafka-cluster:ReadData",
                        "kafka-cluster:DescribeTopic",
                        "kafka-cluster:Connect",
                    ],
                    "Resource": [
                        msk_cluster_arn,
                        msk_topic_arn,
                    ],
                },
                {
                    "Sid": "MSKPolicy",
                    "Effect": "Allow",
                    "Action": ["kafka:GetBootstrapBrokers"],
                    "Resource": "*",
                },
            ],
        }
        role_policy = iam.put_role_policy(
            RoleName=role_name,
            PolicyName=args.iam_policy,
            PolicyDocument=json.dumps(policy),
        )

    role_arn = role["Role"]["Arn"]
    if args.serverless:
        namespace = redshift_serverless.get_namespace(namespaceName=args.name)
        if namespace is None:
            raise ValueError(f"Serverless Redshift namespace {args.name} does not exist")

        existing_roles = [re.search(r"iamRoleArn=(.*)(,|\))", d).group(1) for d in namespace["namespace"]["iamRoles"]]
        if role_arn not in existing_roles:
            _logger.info(
                "Adding IAM Role %s to Serverless Redshift namespace %s",
                role_arn,
                args.name,
            )
            redshift_serverless.update_namespace(namespaceName=args.name, iamRoles=existing_roles + [role_arn])
    else:
        raise NotImplementedError("Only Serverless Redshift is currently supported")

    #    if args.serverless:
    #        workgroup = None
    #        next_token = {}
    #        while workgroup is None:
    #            workgroups = _redshift_serverless.list_workgroups(**next_token)
    #            for wg in workgroups["workgroups"]:
    #                if wg["namespaceName"] == args.name:
    #                    workgroup = wg
    #                    break
    #
    #            if "nextToken" in workgroups:
    #                next_token = {"nextToken": workgroups["nextToken"]}
    #            else:
    #                break
    #        print(workgroup)
    #
    #        def get_credentials(database=None):
    #            kwargs = {}
    #            if database:
    #                kwargs["dbName"] = database
    #            return _redshift_serverless.get_credentials(workgroupName=args.name, **kwargs)
    #
    #    else:
    #        raise NotImplementedError("Only Serverless Redshift is currently supported")

    login_kwargs = {"org_name": args.org_name, "disable_cache": True} if args.org_name else {}
    login(**login_kwargs)

    resp = log_conn().get(
        "/dw-test",
        params={"iam_role": role["Role"]["Arn"], "msk_cluster_arn": msk_cluster_arn},
    )
    resp.raise_for_status()
    _logger.info(f"Finished setting up Redshift: {resp.json()}")
````

## File: py/src/braintrust/cli/__main__.py
````python
import argparse
import logging
import os
import sys
import textwrap

from . import eval, install


def main(args=None):
    """The main routine."""

    # Add the current working directory to sys.path, similar to python's
    # unittesting frameworks.
    sys.path.insert(0, os.getcwd())

    if args is None:
        args = sys.argv[1:]

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Include additional details, including full stack traces on errors.",
    )

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """braintrust is a cli tool to work with Braintrust.
    To see help for a specific subcommand, run `braintrust <subcommand> --help`,
    e.g. `braintrust eval --help`"""
        )
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subcommand", required=True)

    for module in [eval, install]:
        module.build_parser(subparsers, parent_parser)

    args = parser.parse_args(args=args)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s]: %(message)s", level=level)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
````

## File: py/src/braintrust/cli/eval.py
````python
import asyncio
import fnmatch
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from threading import Lock
from typing import List

from .. import login
from ..framework import (
    Evaluator,
    _evals,
    _set_lazy_load,
    bcolors,
    init_experiment,
    parse_filters,
    report_evaluator_result,
    run_evaluator,
)

INCLUDE = [
    "**/eval_*.py",
]
EXCLUDE = ["**/site-packages/**"]

_logger = logging.getLogger("braintrust.eval")


_import_lock = Lock()


@dataclass
class FileHandle:
    in_file: str

    def rebuild(self):
        in_file = os.path.abspath(self.in_file)

        with _import_lock:
            with _set_lazy_load(True):
                _evals.clear()

                try:
                    # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
                    spec = importlib.util.spec_from_file_location("eval", in_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    ret = _evals.copy()
                finally:
                    _evals.clear()

        return ret

    def watch(self):
        raise NotImplementedError


@dataclass
class EvaluatorOpts:
    verbose: bool
    no_send_logs: bool
    no_progress_bars: bool
    terminate_on_failure: bool
    watch: bool
    filters: List[str]


@dataclass
class LoadedEvaluator:
    handle: FileHandle
    evaluator: Evaluator


def update_evaluators(evaluators, handles, terminate_on_failure):
    for handle in handles:
        try:
            module_evals = handle.rebuild()
        except Exception as e:
            if terminate_on_failure:
                raise
            else:
                print(f"Failed to import {handle.in_file}: {e}", file=sys.stderr)
                continue

        for eval_name, evaluator in module_evals.items():
            if not isinstance(evaluator, Evaluator):
                continue

            if eval_name in evaluators:
                _logger.warning(
                    f"Evaluator {eval_name} already exists (in {evaluators[eval_name].handle.in_file} and {handle.in_file}). Will skip {eval_name} in {handle.in_file}."
                )
                continue

            evaluators[eval_name] = LoadedEvaluator(evaluator=evaluator, handle=handle)


async def run_evaluator_task(evaluator, position, opts: EvaluatorOpts):
    experiment = None
    if not opts.no_send_logs:
        experiment = init_experiment(evaluator.project_name, evaluator.metadata)

    try:
        return await run_evaluator(
            experiment, evaluator, position if not opts.no_progress_bars else None, opts.filters
        )
    finally:
        if experiment:
            experiment.close()


async def run_once(handles, evaluator_opts):
    evaluators = {}
    update_evaluators(evaluators, handles, terminate_on_failure=evaluator_opts.terminate_on_failure)

    eval_promises = [
        asyncio.create_task(run_evaluator_task(evaluator.evaluator, idx, evaluator_opts))
        for idx, evaluator in enumerate(evaluators.values())
    ]
    eval_results = [await p for p in eval_promises]

    for eval_name, (results, summary) in zip(evaluators.keys(), eval_results):
        report_evaluator_result(eval_name, results, summary, evaluator_opts.verbose)


def check_match(path_input, include_patterns, exclude_patterns):
    p = os.path.abspath(path_input)
    if include_patterns:
        include = False
        for pattern in include_patterns:
            if fnmatch.fnmatch(p, pattern):
                include = True
                break
        if not include:
            return False

    if exclude_patterns:
        exclude = False
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(p, pattern):
                exclude = True
                break
        return not exclude

    return True


def collect_files(input_path):
    if os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                fname = os.path.join(root, file)
                if check_match(fname, INCLUDE, EXCLUDE):
                    yield fname
    else:
        if check_match(input_path, INCLUDE, EXCLUDE):
            yield input_path


def initialize_handles(files):
    input_paths = files if len(files) > 0 else ["."]

    fnames = set()
    for path in input_paths:
        for fname in collect_files(path):
            fnames.add(os.path.abspath(fname))

    return [FileHandle(in_file=fname) for fname in fnames]


def run(args):
    evaluator_opts = EvaluatorOpts(
        verbose=args.verbose,
        no_send_logs=args.no_send_logs,
        no_progress_bars=args.no_progress_bars,
        terminate_on_failure=args.terminate_on_failure,
        watch=args.watch,
        filters=parse_filters(args.filter) if args.filter else [],
    )

    handles = initialize_handles(args.files)

    if not evaluator_opts.no_send_logs:
        login(
            api_key=args.api_key,
            org_name=args.org_name,
            api_url=args.api_url,
        )

    if args.watch:
        print("Watch mode is not yet implemented", file=sys.stderr)
        exit(1)
    else:
        asyncio.run(run_once(handles, evaluator_opts))


def build_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        "eval",
        help="Run evals locally.",
        parents=[parent_parser],
    )

    parser.add_argument(
        "--api-key",
        help="Specify a braintrust api key. If the parameter is not specified, the BRAINTRUST_API_KEY environment variable will be used.",
    )
    parser.add_argument(
        "--org-name",
        help="The name of a specific organization to connect to. This is useful if you belong to multiple.",
    )
    parser.add_argument(
        "--api-url",
        help="Specify a custom braintrust api url. Defaults to https://www.braintrustdata.com. This is only necessary if you are using an experimental version of Braintrust",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch files for changes and rerun evals when changes are detected",
    )
    parser.add_argument(
        "--filter",
        help="Only run evaluators that match these filters. Each filter is a regular expression (https://docs.python.org/3/library/re.html). For example, --filter metadata.priority='^P0$' input.name='foo.*bar' will only run evaluators that have metadata.priority equal to 'P0' and input.name matching the regular expression 'foo.*bar'.",
        nargs="*",
    )
    parser.add_argument(
        "--no-send-logs",
        action="store_true",
        help="Do not send logs to Braintrust. Useful for testing evaluators without uploading results.",
    )
    parser.add_argument(
        "--no-progress-bars",
        action="store_true",
        help="Do not show progress bars when processing evaluators.",
    )
    parser.add_argument(
        "--terminate-on-failure",
        action="store_true",
        help="If provided, terminates on a failing eval, instead of the default (moving onto the next one).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="A list of files or directories to run. If no files are specified, the current directory is used.",
    )

    parser.set_defaults(func=run)
````

## File: py/src/braintrust/__init__.py
````python
"""
A Python library for logging data to Braintrust. `braintrust` is distributed as
a [library on PyPI](https://pypi.org/project/braintrust/).

### Quickstart

Install the library with pip.

```bash
pip install braintrust
```

Then, run a simple experiment with the following code (replace `YOUR_API_KEY` with
your Braintrust API key):

```python
import braintrust

experiment = braintrust.init(project="PyTest", api_key="YOUR_API_KEY")
experiment.log(
    inputs={"test": 1},
    output="foo",
    expected="bar",
    scores={
        "n": 0.5,
    },
    metadata={
        "id": 1,
    },
)
print(experiment.summarize())
```

### API Reference
"""

from .framework import *
from .logger import *
from .oai import wrap_openai
````

## File: py/src/braintrust/aws.py
````python
import sys
from functools import cached_property

import boto3


class LazyClient:
    def __init__(self, client_name):
        self.client_name = client_name
        self.client = None

    def __getattr__(self, name):
        if self.client is None:
            self.client = boto3.client(self.client_name)
        return getattr(self.client, name)


def __getattr__(name: str):
    return LazyClient(name.replace("_", "-"))
````

## File: py/src/braintrust/cache.py
````python
from pathlib import Path

CACHE_PATH = Path.home() / ".cache" / "braintrust"
EXPERIMENTS_PATH = CACHE_PATH / "experiments"
LOGIN_INFO_PATH = CACHE_PATH / "api_info.json"
````

## File: py/src/braintrust/framework.py
````python
import abc
import asyncio
import dataclasses
import inspect
import json
import re
import traceback
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, TypeVar, Union

from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm as std_tqdm

from autoevals import Score, Scorer

from .logger import NOOP_SPAN, Span, current_span, start_span
from .logger import init as _init_experiment
from .util import SerializableDataClass

Metadata = Dict[str, Any]
Input = TypeVar("Input")
Output = TypeVar("Output")


# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclasses.dataclass
class EvalCase(SerializableDataClass):
    """
    An evaluation case. This is a single input to the evaluation task, along with an optional expected
    output and metadata.
    """

    input: Input
    expected: Optional[Output] = None
    metadata: Optional[Metadata] = None


class EvalHooks(abc.ABC):
    """
    An object that can be used to add metadata to an evaluation. This is passed to the `task` function.
    """

    @property
    @abc.abstractmethod
    def span(self) -> Span:
        """
        Access the span under which the task is run. Also accessible via braintrust.current_span()
        """

    @abc.abstractmethod
    def meta(self, **info) -> None:
        """
        Adds metadata to the evaluation. This metadata will be logged to the Braintrust. You can pass in metadaa
        as keyword arguments, e.g. `hooks.meta(foo="bar")`.
        """
        ...


class EvalScorerArgs(SerializableDataClass):
    """
    Arguments passed to an evaluator scorer. This includes the input, expected output, actual output, and metadata.
    """

    input: Input
    output: Output
    expected: Optional[Output] = None
    metadata: Optional[Metadata] = None


EvalScorer = Union[
    Scorer,
    Callable[[Input, Output, Output], Score],
    Callable[[Input, Output, Output], Awaitable[Score]],
]


@dataclasses.dataclass
class EvalMetadata(SerializableDataClass):
    """
    Additional metadata for the eval definition, such as experiment name.
    """

    """
    Specify a name for the experiment holding the eval results.
    """
    experiment_name: Optional[str] = None


def eval_metadata_to_init_options(metadata: Optional[EvalMetadata] = None) -> Dict:
    if metadata is None:
        return dict()
    return dict(experiment=metadata.experiment_name)


@dataclasses.dataclass
class Evaluator:
    """
    An evaluator is an abstraction that defines an evaluation dataset, a task to run on the dataset, and a set of
    scorers to evaluate the results of the task. Each method attribute can be synchronous or asynchronous (for
    optimal performance, it is recommended to provide asynchronous implementations).

    You should not create Evaluators directly if you plan to use the Braintrust eval framework. Instead, you should
    create them using the `Eval()` method, which will register them so that `braintrust eval ...` can find them.
    """

    """
    The name of the project the eval falls under.
    """
    project_name: str

    """
    A name that uniquely defines this type of experiment. You do not need to change it each time the experiment runs, but you should not have other experiments in your code with the same name.
    """
    eval_name: str

    """
    Returns an iterator over the evaluation dataset. Each element of the iterator should be an `EvalCase` or a dict
    with the same fields as an `EvalCase` (`input`, `expected`, `metadata`).
    """
    data: Union[
        Iterator[EvalCase],
        Awaitable[Iterator[EvalCase]],
        Callable[[], Union[Iterator[EvalCase], Awaitable[Iterator[EvalCase]]]],
    ]

    """
    Runs the evaluation task on a single input. The `hooks` object can be used to add metadata to the evaluation.
    """
    task: Union[
        Callable[[Input, EvalHooks], Union[Output, Awaitable[Output]]],
        Callable[[Input], Union[Output, Awaitable[Output]]],
    ]

    """
    A list of scorers to evaluate the results of the task. Each scorer can be a Scorer object or a function
    that takes `input`, `output`, and `expected` arguments and returns a `Score` object. The function can be async.
    """
    scores: List[EvalScorer]

    """
    Optional additional metadata for the eval definition, such as experiment name.
    """
    metadata: Optional[EvalMetadata]


_evals = {}
_lazy_load = False


@contextmanager
def _set_lazy_load(lazy_load: bool):
    global _lazy_load
    current = _lazy_load
    try:
        _lazy_load = lazy_load
        yield
    finally:
        _lazy_load = current


def pluralize(n, singular, plural):
    if n == 1:
        return singular
    else:
        return plural


def report_evaluator_result(eval_name, results, summary, verbose):
    failing_results = [x for x in results if x.error]
    if len(failing_results) > 0:
        print(
            f"{bcolors.FAIL}Evaluator {eval_name} failed with {len(failing_results)} {pluralize(len(failing_results), 'error', 'errors')}{bcolors.ENDC}"
        )

        for result in failing_results:
            info = "".join(
                result.exc_info if verbose else traceback.format_exception_only(type(result.error), result.error)
            ).rstrip()
            print(f"{bcolors.FAIL}{info}{bcolors.ENDC}")
    if summary:
        print(f"{summary}")
    else:
        scores_by_name = defaultdict(lambda: (0, 0))
        for result in results:
            for name, score in result.scores.items():
                curr = scores_by_name[name]
                scores_by_name[name] = (curr[0] + score, curr[1] + 1)

        print(f"Average scores for {eval_name}:")
        for name, (total, count) in scores_by_name.items():
            print(f"  {name}: {total / count}")


def _make_eval_name(name: str, metadata: Optional[EvalMetadata]):
    out = name
    if metadata is not None and metadata.experiment_name is not None:
        out += f" [experiment_name={metadata.experiment_name}]"
    return out


def Eval(
    name: str,
    data: Callable[[], Union[Iterator[EvalCase], AsyncIterator[EvalCase]]],
    task: Callable[[Input, EvalHooks], Union[Output, Awaitable[Output]]],
    scores: List[EvalScorer],
    metadata: Union[Optional[EvalMetadata], Dict] = None,
):
    """
    A function you can use to define an evaluator. This is a convenience wrapper around the `Evaluator` class.

    Example:
    ```python
    Eval(
        name="my-evaluator",
        data=lambda: [
            EvalCase(input=1, expected=2),
            EvalCase(input=2, expected=4),
        ],
        task=lambda input, hooks: input * 2,
        scores=[
            NumericDiff,
        ],
    )
    ```

    :param name: The name of the evaluator. This corresponds to a project name in Braintrust.
    :param data: Returns an iterator over the evaluation dataset. Each element of the iterator should be a `EvalCase`.
    :param task: Runs the evaluation task on a single input. The `hooks` object can be used to add metadata to the evaluation.
    :param scores: A list of scorers to evaluate the results of the task. Each scorer can be a Scorer object or a function
    that takes an `EvalScorerArgs` object and returns a `Score` object.
    :param metadata: Optional additional metadata for the eval definition, such as experiment name.
    :return: An `Evaluator` object.
    """
    if isinstance(metadata, dict):
        metadata = EvalMetadata(**metadata)

    eval_name = _make_eval_name(name, metadata)

    global _evals
    if eval_name in _evals:
        raise ValueError(f"Evaluator {eval_name} already exists")

    evaluator = Evaluator(
        eval_name=eval_name, project_name=name, data=data, task=task, scores=scores, metadata=metadata
    )

    if _lazy_load:
        _evals[eval_name] = evaluator
    else:
        # https://stackoverflow.com/questions/55409641/asyncio-run-cannot-be-called-from-a-running-event-loop-when-using-jupyter-no
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'RuntimeError: There is no current event loop...'
            loop = None

        async def run_to_completion():
            with init_experiment(evaluator.project_name, evaluator.metadata) as experiment:
                results, summary = await run_evaluator(experiment, evaluator, 0, [])
                report_evaluator_result(evaluator.eval_name, results, summary, True)

        if loop:
            return loop.create_task(run_to_completion())
        else:
            asyncio.run(run_to_completion())


@dataclasses.dataclass
class Filter:
    path: List[str]
    pattern: re.Pattern


def serialize_json_with_plain_string(v: Any) -> str:
    if isinstance(v, str):
        return v
    else:
        return json.dumps(v)


def deserialize_plain_string_as_json(s: str) -> Any:
    try:
        return {"value": json.loads(s)}
    except json.JSONDecodeError as e:
        return {"value": s, "error": e}


def parse_filters(filters: List[str]) -> List[Filter]:
    result = []
    for f in filters:
        equals_idx = f.index("=")
        if equals_idx == -1:
            raise ValueError(f"Invalid filter {f}")
        path, value = f[:equals_idx], f[equals_idx + 1 :]
        deserialized_value = deserialize_plain_string_as_json(value)["value"]
        if not isinstance(deserialized_value, str):
            deserialized_value = value
        result.append(
            Filter(
                path=path.split("."),
                pattern=re.compile(deserialized_value),
            )
        )

    return result


def evaluate_filter(object, filter: Filter):
    key = object
    for p in filter.path:
        key = key.get(p)
        if key is None:
            return False
    return filter.pattern.match(serialize_json_with_plain_string(key)) is not None


@dataclasses.dataclass
class EvalResult:
    output: Output
    metadata: Metadata
    scores: Dict[str, Score]
    error: Optional[Exception] = None
    exc_info: Optional[str] = None


class DictEvalHooks(EvalHooks):
    def __init__(self, metadata):
        self.metadata = metadata
        self._span = None

    @property
    def span(self):
        return self._span

    def set_span(self, span):
        self._span = span

    def meta(self, **info):
        self.metadata.update(info)


def init_experiment(project_name, metadata):
    ret = _init_experiment(project_name, **eval_metadata_to_init_options(metadata))
    summary = ret.summarize(summarize_scores=False)
    print(f"Experiment {ret.name} is running at {summary.experiment_url}")
    return ret


async def run_evaluator(experiment, evaluator: Evaluator, position: Optional[int], filters: List[Filter]):
    #   if (typeof evaluator.data === "string") {
    #     throw new Error("Unimplemented: string data paths");
    #   }
    #   const dataResult = evaluator.data();
    #   let data = null;
    #   if (dataResult instanceof Promise) {
    #     data = await dataResult;
    #   } else {
    #     data = dataResult;
    #   }

    async def await_or_run(f, *args, **kwargs):
        if inspect.iscoroutinefunction(f):
            return await f(*args, **kwargs)
        else:
            return f(*args, **kwargs)

    async def await_or_run_scorer(scorer, scorer_idx, **kwargs):
        name = scorer._name() if hasattr(scorer, "_name") else scorer.__name__
        if name == "<lambda>":
            name = f"scorer_{scorer_idx}"
        with start_span(name=name, input=dict(**kwargs)):
            score = scorer.eval_async if isinstance(scorer, Scorer) else scorer

            scorer_args = kwargs

            signature = inspect.signature(score)
            scorer_accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            if not scorer_accepts_kwargs:
                scorer_args = {k: v for k, v in scorer_args.items() if k in signature.parameters}

            result = await await_or_run(score, **scorer_args)
            if isinstance(result, Score):
                result_rest = result.as_dict()
                result_metadata = result_rest.pop("metadata", {})
                current_span().log(output=result_rest, metadata=result_metadata)
            else:
                current_span().log(output=result)
            return result

    async def run_evaluator_task(datum):
        if isinstance(datum, dict):
            datum = EvalCase(**datum)

        metadata = {**(datum.metadata or {})}
        output = None
        error = None
        exc_info = None
        scores = {}

        if experiment:
            root_span = experiment.start_span("eval", input=datum.input, expected=datum.expected)
        else:
            root_span = NOOP_SPAN
        with root_span:
            try:
                hooks = DictEvalHooks(metadata)

                # Check if the task takes a hooks argument
                task_args = [datum.input]
                if len(inspect.signature(evaluator.task).parameters) == 2:
                    task_args.append(hooks)

                with current_span().start_span("task") as task_span:
                    hooks.set_span(task_span)
                    output = await await_or_run(evaluator.task, *task_args)
                    task_span.log(input=task_args[0], output=output)
                current_span().log(output=output)

                # First, resolve the scorers if they are classes
                scorers = [
                    scorer() if inspect.isclass(scorer) and issubclass(scorer, Scorer) else scorer
                    for scorer in evaluator.scores
                ]
                score_promises = [
                    asyncio.create_task(await_or_run_scorer(score, idx, **datum.as_dict(), output=output))
                    for idx, score in enumerate(scorers)
                ]
                score_results = [await p for p in score_promises]
                score_metadata = {}
                for scorer, score_result in zip(scorers, score_results):
                    if not isinstance(score_result, Score):
                        score_result = Score(name=scorer.__name__, score=score_result)
                    scores[score_result.name] = score_result.score
                    m = {**(score_result.metadata or {})}
                    if score_result.error is not None:
                        m["error"] = str(score_result.error)
                    if len(m) > 0:
                        score_metadata[score_result.name] = m

                if len(score_metadata) > 0:
                    hooks.meta(scores=score_metadata)

                # XXX: We could probably log these as they are being produced
                current_span().log(metadata=metadata, scores=scores)
            except Exception as e:
                error = e
                # Python3.10 has a different set of arguments to format_exception than earlier versions,
                # so just capture the stack trace here.
                exc_info = traceback.format_exc()

        return EvalResult(output=output, metadata=metadata, scores=scores, error=error, exc_info=exc_info)

    data_iterator = evaluator.data
    if inspect.isfunction(data_iterator):
        data_iterator = data_iterator()

    if not inspect.isasyncgen(data_iterator):

        async def to_async(it):
            for d in it:
                yield d

        data_iterator = to_async(data_iterator)

    async def filtered_iterator(it):
        async for datum in it:
            if all(evaluate_filter(datum, f) for f in filters):
                yield datum

    tasks = []
    with async_tqdm(
        filtered_iterator(data_iterator),
        desc=f"{evaluator.eval_name} (data)",
        position=position,
        disable=position is None,
    ) as pbar:
        async for datum in pbar:
            tasks.append(asyncio.create_task(run_evaluator_task(datum)))

    results = []
    for task in std_tqdm(tasks, desc=f"{evaluator.eval_name} (tasks)", position=position, disable=position is None):
        results.append(await task)

    summary = experiment.summarize() if experiment else None
    return results, summary


__all__ = ["Evaluator", "Eval", "Score", "EvalCase", "EvalHooks"]
````

## File: py/src/braintrust/gitutil.py
````python
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from functools import cache as _cache
from typing import Optional

from .util import SerializableDataClass

# https://stackoverflow.com/questions/48399498/git-executable-not-found-in-python
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

_logger = logging.getLogger("braintrust.gitutil")
_gitlock = threading.RLock()


@dataclass
class RepoStatus(SerializableDataClass):
    """Information about the current HEAD of the repo."""

    commit: Optional[str]
    branch: Optional[str]
    tag: Optional[str]
    dirty: bool
    author_name: Optional[str]
    author_email: Optional[str]
    commit_message: Optional[str]
    commit_time: Optional[str]


@_cache
def _current_repo():
    try:
        return git.Repo(search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        return None


@_cache
def _get_base_branch(remote=None):
    repo = _current_repo()
    remote = repo.remote(**({} if remote is None else {"name": remote})).name

    # NOTE: This should potentially be configuration that we derive from the project,
    # instead of spending a second or two computing it each time we run an experiment.

    # To speed this up in the short term, we pick from a list of common names
    # and only fall back to the remote origin if required.
    COMMON_BASE_BRANCHES = ["main", "master", "develop"]
    repo_branches = set(b.name for b in repo.branches)
    if sum(b in repo_branches for b in COMMON_BASE_BRANCHES) == 1:
        for b in COMMON_BASE_BRANCHES:
            if b in repo_branches:
                return (remote, b)
        raise RuntimeError("Impossible")

    try:
        s = subprocess.check_output(["git", "remote", "show", "origin"]).decode()
        match = re.search(r"\s*HEAD branch:\s*(.*)$", s, re.MULTILINE)
        if match is None:
            raise RuntimeError("Could not find HEAD branch in remote " + remote)
        branch = match.group(1)
    except Exception as e:
        _logger.warning(f"Could not find base branch for remote {remote}", e)
        branch = "main"
    return (remote, branch)


def _get_base_branch_ancestor(remote=None):
    try:
        remote_name, base_branch = _get_base_branch(remote)
    except Exception as e:
        _logger.warning(
            f"Skipping git metadata. This is likely because the repository has not been published to a remote yet. {e}"
        )
        return None

    head = "HEAD" if _current_repo().is_dirty() else "HEAD^"
    try:
        return subprocess.check_output(["git", "merge-base", head, f"{remote_name}/{base_branch}"]).decode().strip()
    except subprocess.CalledProcessError as e:
        # _logger.warning(f"Could not find a common ancestor with {remote_name}/{base_branch}")
        return None


def get_past_n_ancestors(n=10, remote=None):
    with _gitlock:
        repo = _current_repo()
        if repo is None:
            return

        ancestor_output = _get_base_branch_ancestor()
        if ancestor_output is None:
            return
        ancestor = repo.commit(ancestor_output)
        for _ in range(n):
            yield ancestor.hexsha
            try:
                if ancestor.parents:
                    ancestor = ancestor.parents[0]
                else:
                    break
            except ValueError:
                # Since parents are fetched on-demand, this can happen if the
                # downloaded repo does not have information for this commit's
                # parent.
                break


def attempt(op):
    try:
        return op()
    except TypeError:
        return None
    except git.GitCommandError:
        return None


def get_repo_status():
    with _gitlock:
        repo = _current_repo()
        if repo is None:
            return None

        commit = None
        commit_message = None
        commit_time = None
        author_name = None
        author_email = None
        tag = None
        branch = None

        dirty = repo.is_dirty()

        commit = attempt(lambda: repo.head.commit.hexsha).strip()
        commit_message = attempt(lambda: repo.head.commit.message).strip()
        commit_time = attempt(lambda: repo.head.commit.committed_datetime.isoformat())
        author_name = attempt(lambda: repo.head.commit.author.name).strip()
        author_email = attempt(lambda: repo.head.commit.author.email).strip()
        tag = attempt(lambda: repo.git.describe("--tags", "--exact-match", "--always"))

        branch = attempt(lambda: repo.active_branch.name)

        return RepoStatus(
            commit=commit,
            branch=branch,
            tag=tag,
            dirty=dirty,
            author_name=author_name,
            author_email=author_email,
            commit_message=commit_message,
            commit_time=commit_time,
        )
````

## File: py/src/braintrust/logger.py
````python
import atexit
import concurrent.futures
import contextvars
import dataclasses
import datetime
import inspect
import json
import logging
import os
import queue
import sys
import textwrap
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from functools import partial, wraps
from getpass import getpass
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cache import CACHE_PATH, EXPERIMENTS_PATH, LOGIN_INFO_PATH
from .gitutil import get_past_n_ancestors, get_repo_status
from .merge_row_batch import merge_row_batch
from .resource_manager import ResourceManager
from .util import (
    GLOBAL_PROJECT,
    IS_MERGE_FIELD,
    TRANSACTION_ID_FIELD,
    AugmentedHTTPError,
    SerializableDataClass,
    encode_uri_component,
    get_caller_location,
    merge_dicts,
    response_raise_for_status,
)


class Span(ABC):
    """
    A Span encapsulates logged data and metrics for a unit of work. This interface is shared by all span implementations.

    We suggest using one of the various `startSpan` methods, instead of creating Spans directly. See `Span.startSpan` for full details.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Row ID of the span."""

    @property
    @abstractmethod
    def span_id(self) -> str:
        """Span ID of the span. This is used to link spans together."""

    @property
    @abstractmethod
    def root_span_id(self) -> str:
        """Span ID of the root span in the full trace."""

    @abstractmethod
    def log(self, **event):
        """Incrementally update the current span with new data. The event will be batched and uploaded behind the scenes.

        :param **event: Data to be logged. See `Experiment.log` for full details.
        """

    @abstractmethod
    def start_span(self, name, span_attributes={}, start_time=None, set_current=None, **event):
        """Create a new span. This is useful if you want to log more detailed trace information beyond the scope of a single log event. Data logged over several calls to `Span.log` will be merged into one logical row.

        We recommend running spans within context managers (`with start_span(...) as span`) to automatically mark them as current and ensure they are terminated. If you wish to start a span outside a callback, be sure to terminate it with `span.end()`.

        :param name: The name of the span.
        :param span_attributes: Optional additional attributes to attach to the span, such as a type name.
        :param start_time: Optional start time of the span, as a timestamp in seconds.
        :param set_current: If true (the default), the span will be marked as the currently-active span for the duration of the context manager. Unless the span is bound to a context manager, it will not be marked as current. Equivalent to calling `with braintrust.with_current(span)`.
        :param **event: Data to be logged. See `Experiment.log` for full details.
        :returns: The newly-created `Span`
        """

    @abstractmethod
    def end(self, end_time=None) -> float:
        """Terminate the span. Returns the end time logged to the row's metrics. After calling end, you may not invoke any further methods on the span object, except for the property accessors.

        Will be invoked automatically if the span is bound to a context manager.

        :param end_time: Optional end time of the span, as a timestamp in seconds.
        :returns: The end time logged to the span metrics.
        """

    @abstractmethod
    def close(self, end_time=None) -> float:
        """Alias for `end`."""

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self):
        pass


# DEVNOTE: This is copied into autoevals/py/autoevals/util.py
class _NoopSpan(Span):
    """A fake implementation of the Span API which does nothing. This can be used as the default span."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def id(self):
        return ""

    @property
    def span_id(self):
        return ""

    @property
    def root_span_id(self):
        return ""

    def log(self, **event):
        pass

    def start_span(self, name, span_attributes={}, start_time=None, set_current=None, **event):
        return self

    def end(self, end_time=None):
        return end_time or time.time()

    def close(self, end_time=None):
        return self.end(end_time)

    def __enter__(self):
        return self

    def __exit__(self, type, value, callback):
        del type, value, callback


NOOP_SPAN = _NoopSpan()


class BraintrustState:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.current_experiment = contextvars.ContextVar("braintrust_current_experiment", default=None)
        self.current_logger = contextvars.ContextVar("braintrust_current_logger", default=None)
        self.current_span = contextvars.ContextVar("braintrust_current_span", default=NOOP_SPAN)

        self.api_url = None
        self.login_token = None
        self.org_id = None
        self.org_name = None
        self.log_url = None
        self.logged_in = False

        self._api_conn = None
        self._log_conn = None
        self._user_info = None

    def api_conn(self):
        if not self._api_conn:
            if not self.api_url:
                raise RuntimeError("Must initialize api_url before requesting api_conn")
            self._api_conn = HTTPConnection(self.api_url)
        return self._api_conn

    def log_conn(self):
        if not self._log_conn:
            if not self.log_url:
                raise RuntimeError("Must initialize log_url before requesting log_conn")
            self._log_conn = HTTPConnection(self.log_url)
        return self._log_conn

    def user_info(self):
        if not self._user_info:
            self._user_info = self.log_conn().get_json("ping")
        return self._user_info

    def set_user_info_if_null(self, info):
        if not self._user_info:
            self._user_info = info


_state = BraintrustState()
_logger = logging.getLogger("braintrust")


class _UnterminatedObjectsHandler:
    """A utility to keep track of objects that should be cleaned up before program exit. At the end of the program, the _UnterminatedObjectsHandler will print out all un-terminated objects as a warning."""

    def __init__(self):
        self._unterminated_objects = ResourceManager({})
        atexit.register(self._warn_unterminated)

    def add_unterminated(self, obj, created_location=None):
        with self._unterminated_objects.get() as unterminated_objects:
            unterminated_objects[obj] = created_location

    def remove_unterminated(self, obj):
        with self._unterminated_objects.get() as unterminated_objects:
            del unterminated_objects[obj]

    def _warn_unterminated(self):
        with self._unterminated_objects.get() as unterminated_objects:
            if not unterminated_objects:
                return
            warning_message = "WARNING: Did not close the following braintrust objects. We recommend running `.close` on the listed objects, or binding them to a context manager so they are closed automatically:"
            for obj, created_location in unterminated_objects.items():
                msg = f"\n\tObject of type {type(obj)}"
                if created_location:
                    msg += f" created at {created_location}"
                warning_message += msg
            print(warning_message, file=sys.stderr)


_unterminated_objects = _UnterminatedObjectsHandler()


class HTTPConnection:
    def __init__(self, base_url):
        self.base_url = base_url
        self.token = None

        self._reset(total=0)

    def ping(self):
        try:
            resp = self.get("ping")
            _state.set_user_info_if_null(resp.json())
            return resp.ok
        except requests.exceptions.ConnectionError:
            return False

    def make_long_lived(self):
        # Following a suggestion in https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
        self._reset(connect=10, backoff_factor=0.5)

    @staticmethod
    def sanitize_token(token):
        return token.rstrip("\n")

    def set_token(self, token):
        token = HTTPConnection.sanitize_token(token)
        self.token = token
        self._set_session_token()

    def _reset(self, **retry_kwargs):
        self.session = requests.Session()

        retry = Retry(**retry_kwargs)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._set_session_token()

    def _set_session_token(self):
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def get(self, path, *args, **kwargs):
        return self.session.get(_urljoin(self.base_url, path), *args, **kwargs)

    def post(self, path, *args, **kwargs):
        return self.session.post(_urljoin(self.base_url, path), *args, **kwargs)

    def delete(self, path, *args, **kwargs):
        return self.session.delete(_urljoin(self.base_url, path), *args, **kwargs)

    def get_json(self, object_type, args=None, retries=0):
        tries = retries + 1
        for i in range(tries):
            resp = self.get(f"/{object_type}", params=args)
            if i < tries - 1 and not resp.ok:
                _logger.warning(f"Retrying API request {object_type} {args} {resp.status_code} {resp.text}")
                continue
            response_raise_for_status(resp)

            return resp.json()

    def post_json(self, object_type, args):
        resp = self.post(f"/{object_type.lstrip('/')}", json=args)
        response_raise_for_status(resp)
        return resp.json()


# Sometimes we'd like to launch network requests concurrently. We provide a
# thread pool to accomplish this. Use a multiple of number of CPU cores to limit
# concurrency.
HTTP_REQUEST_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count())


def log_conn():
    return _state.log_conn()


def api_conn():
    return _state.api_conn()


def user_info():
    return _state.user_info()


def org_id():
    return _state.org_id


class ModelWrapper:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name: str) -> Any:
        return self.data[name]


# 6 MB (from our own testing).
MAX_REQUEST_SIZE = 6 * 1024 * 1024


def construct_json_array(items):
    return "[" + ",".join(items) + "]"


DEFAULT_BATCH_SIZE = 100
NUM_RETRIES = 3


class _LogThread:
    def __init__(self, name=None):
        self.flush_lock = threading.RLock()
        self.start_thread_lock = threading.RLock()
        self.thread = threading.Thread(target=self._publisher, daemon=True)
        self.started = False

        log_namespace = "braintrust"
        if name:
            log_namespace += f" [{name}]"

        self.logger = logging.getLogger(log_namespace)

        try:
            queue_size = int(os.environ.get("BRAINTRUST_QUEUE_SIZE"))
        except Exception:
            queue_size = 1000
        self.queue = queue.Queue(maxsize=queue_size)
        # Each time we put items in the queue, we increment a semaphore to
        # indicate to any consumer thread that it should attempt a flush.
        self.queue_filled_semaphore = threading.Semaphore(value=0)

        atexit.register(self._finalize)

    def log(self, *args):
        self._start()
        for event in args:
            try:
                _ = json.dumps(event)
            except TypeError as e:
                raise Exception(f"All logged values must be JSON-serializable: {event}") from e

            try:
                self.queue.put_nowait(event)
            except queue.Full:
                # Notify consumers to start draining the queue.
                self.queue_filled_semaphore.release()
                self.queue.put(event)
        self.queue_filled_semaphore.release()

    def _start(self):
        # Double read to avoid contention in the common case.
        if not self.started:
            with self.start_thread_lock:
                if not self.started:
                    self.thread.start()
                    self.started = True

    def _finalize(self):
        self.logger.info("Flushing final log events...")
        self.flush()

    def _publisher(self, batch_size=None):
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        while True:
            # Wait for some data on the queue before trying to flush.
            self.queue_filled_semaphore.acquire()
            try:
                self.flush(**kwargs)
            except Exception:
                traceback.print_exc()

    def flush(self, batch_size=100):
        # We cannot have multiple threads flushing in parallel, because the
        # order of published elements would be undefined.
        with self.flush_lock:
            # Drain the queue.
            all_items = []
            try:
                for _ in range(self.queue.qsize()):
                    all_items.append(self.queue.get_nowait())
            except queue.Empty:
                pass
            all_items = list(reversed(merge_row_batch(all_items)))

            if len(all_items) == 0:
                return

            conn = _state.log_conn()
            post_promises = []
            while True:
                items = []
                items_len = 0
                while len(items) < batch_size and items_len < MAX_REQUEST_SIZE / 2:
                    if len(all_items) > 0:
                        item = all_items.pop()
                    else:
                        break

                    item_s = json.dumps(item)
                    items.append(item_s)
                    items_len += len(item_s)

                if len(items) == 0:
                    break

                try:
                    post_promises.append(HTTP_REQUEST_THREAD_POOL.submit(_LogThread._submit_logs_request, items, conn))
                except RuntimeError:
                    # If the thread pool has shut down, e.g. because the process is terminating, run the request the
                    # old fashioned way.
                    _LogThread._submit_logs_request(items, conn)

            concurrent.futures.wait(post_promises)

    @staticmethod
    def _submit_logs_request(items, conn):
        items_s = construct_json_array(items)
        for i in range(NUM_RETRIES):
            start_time = time.time()
            resp = conn.post("/logs", data=items_s)
            if resp.ok:
                return
            retrying_text = "" if i + 1 == NUM_RETRIES else " Retrying"
            _logger.warning(
                f"log request failed. Elapsed time: {time.time() - start_time} seconds. Payload size: {len(items_s)}. Error: {resp.status_code}: {resp.text}.{retrying_text}"
            )
        if not resp.ok:
            _logger.warning(f"log request failed after {NUM_RETRIES} retries. Dropping batch")


def _ensure_object(object_type, object_id, force=False):
    experiment_path = EXPERIMENTS_PATH / f"{object_id}.parquet"

    if force or not experiment_path.exists():
        os.makedirs(EXPERIMENTS_PATH, exist_ok=True)
        conn = _state.log_conn()
        resp = conn.get(
            f"object/{object_type}",
            params={"id": object_id},
            headers={
                "Accept": "application/octet-stream",
            },
        )

        with open(experiment_path, "wb") as f:
            f.write(resp.content)

    return experiment_path


def init(
    project: str,
    experiment: str = None,
    description: str = None,
    dataset: "Dataset" = None,
    update: bool = False,
    base_experiment: str = None,
    is_public: bool = False,
    api_url: str = None,
    api_key: str = None,
    org_name: str = None,
    disable_cache: bool = False,
    set_current: bool = None,
):
    """
    Log in, and then initialize a new experiment in a specified project. If the project does not exist, it will be created.

    Remember to close your experiment when it is finished by calling `Experiment.close`. We recommend binding the experiment to a context manager (`with braintrust.init(...) as experiment`) to automatically mark it as current and ensure it is terminated.

    :param project: The name of the project to create the experiment in.
    :param experiment: The name of the experiment to create. If not specified, a name will be generated automatically.
    :param description: (Optional) An optional description of the experiment.
    :param dataset: (Optional) A dataset to associate with the experiment. The dataset must be initialized with `braintrust.init_dataset` before passing
    it into the experiment.
    :param update: If the experiment already exists, continue logging to it.
    :param base_experiment: An optional experiment name to use as a base. If specified, the new experiment will be summarized and compared to this
    experiment. Otherwise, it will pick an experiment by finding the closest ancestor on the default (e.g. main) branch.
    :param is_public: An optional parameter to control whether the experiment is publicly visible to anybody with the link or privately visible to only members of the organization. Defaults to private.
    :param api_url: The URL of the Braintrust API. Defaults to https://www.braintrustdata.com.
    :param api_key: The API key to use. If the parameter is not specified, will try to use the `BRAINTRUST_API_KEY` environment variable. If no API
    key is specified, will prompt the user to login.
    :param org_name: (Optional) The name of a specific organization to connect to. This is useful if you belong to multiple.
    :param disable_cache: Do not use cached login information.
    :param set_current: If true (default), set the currently-active experiment to the newly-created one. Unless the experiment is bound to a context manager, it will not be marked as current. Equivalent to calling `with braintrust.with_current(experiment)`.
    :returns: The experiment object.
    """
    login(org_name=org_name, disable_cache=disable_cache, api_key=api_key, api_url=api_url)
    return Experiment(
        project_name=project,
        experiment_name=experiment,
        description=description,
        dataset=dataset,
        update=update,
        base_experiment=base_experiment,
        is_public=is_public,
        set_current=set_current,
    )


def init_dataset(
    project: str,
    name: str = None,
    description: str = None,
    version: "str | int" = None,
    api_url: str = None,
    api_key: str = None,
    org_name: str = None,
    disable_cache: bool = False,
):
    """
    Create a new dataset in a specified project. If the project does not exist, it will be created.

    Remember to close your dataset when it is finished by calling `Dataset.close`. We recommend wrapping the dataset within a context manager (`with braintrust.init_dataset(...) as dataset`) to ensure it is terminated.

    :param project: The name of the project to create the dataset in.
    :param name: The name of the dataset to create. If not specified, a name will be generated automatically.
    :param description: An optional description of the dataset.
    :param version: An optional version of the dataset (to read). If not specified, the latest version will be used.
    :param api_url: The URL of the Braintrust API. Defaults to https://www.braintrustdata.com.
    :param api_key: The API key to use. If the parameter is not specified, will try to use the `BRAINTRUST_API_KEY` environment variable. If no API
    key is specified, will prompt the user to login.
    :param org_name: (Optional) The name of a specific organization to connect to. This is useful if you belong to multiple.
    :param disable_cache: Do not use cached login information.
    :returns: The dataset object.
    """
    login(org_name=org_name, disable_cache=disable_cache, api_key=api_key, api_url=api_url)

    return Dataset(
        project_name=project,
        name=name,
        description=description,
        version=version,
    )


def init_logger(
    project: str = None,
    project_id: str = None,
    async_flush: bool = True,
    set_current: bool = True,
    api_url: str = None,
    api_key: str = None,
    org_name: str = None,
    disable_cache: bool = False,
):
    """
    Create a new logger in a specified project. If the project does not exist, it will be created.

    :param project: The name of the project to log into. If unspecified, will default to the Global project.
    :param project_id: The id of the project to log into. This takes precedence over project if specified.
    :param async_flush: If true (the default), log events will be batched and sent asynchronously in a background thread. If false, log events will be sent synchronously. Set to false in serverless environments.
    :param set_current: If true (default), set the currently-active logger to the newly-created one. Unless the logger is bound to a context manager, it will not be marked as current. Equivalent to calling `with braintrust.with_current(logger)`.
    :param api_url: The URL of the Braintrust API. Defaults to https://www.braintrustdata.com.
    :param api_key: The API key to use. If the parameter is not specified, will try to use the `BRAINTRUST_API_KEY` environment variable. If no API
    key is specified, will prompt the user to login.
    :param org_name: (Optional) The name of a specific organization to connect to. This is useful if you belong to multiple.
    :param disable_cache: Do not use cached login information.
    :returns: The newly created Logger.
    """

    def lazy_login():
        login(org_name=org_name, disable_cache=disable_cache, api_key=api_key, api_url=api_url)

    return Logger(
        lazy_login=lazy_login,
        project=Project(name=project, id=project_id),
        async_flush=async_flush,
        set_current=set_current,
    )


login_lock = threading.RLock()


def login(api_url=None, api_key=None, org_name=None, disable_cache=False, force_login=False):
    """
    Log into Braintrust. This will prompt you for your API token, which you can find at
    https://www.braintrustdata.com/app/token. This method is called automatically by `init()`.

    :param api_url: The URL of the Braintrust API. Defaults to https://www.braintrustdata.com.
    :param api_key: The API key to use. If the parameter is not specified, will try to use the `BRAINTRUST_API_KEY` environment variable. If no API
    key is specified, will prompt the user to login.
    :param org_name: (Optional) The name of a specific organization to connect to. This is useful if you belong to multiple.
    :param disable_cache: Do not use cached login information.
    :param force_login: Login again, even if you have already logged in (by default, this function will exit quickly if you have already logged in)
    """

    global _state

    # Only permit one thread to login at a time
    with login_lock:
        if api_url is None:
            api_url = os.environ.get("BRAINTRUST_API_URL", "https://www.braintrustdata.com")

        if api_key is None:
            api_key = os.environ.get("BRAINTRUST_API_KEY")

        if org_name is None:
            org_name = os.environ.get("BRAINTRUST_ORG_NAME")

        # If any provided login inputs disagree with our existing settings,
        # force login.
        if (
            api_url != _state.api_url
            or (api_key is not None and HTTPConnection.sanitize_token(api_key) != _state.login_token)
            or (org_name is not None and org_name != _state.org_name)
        ):
            force_login = True

        if not force_login and _state.logged_in:
            # We have already logged in
            return

        _state = BraintrustState()

        _state.api_url = api_url

        os.makedirs(CACHE_PATH, exist_ok=True)

        conn = None
        if api_key is not None:
            resp = requests.post(_urljoin(_state.api_url, "/api/apikey/login"), json={"token": api_key})
            if not resp.ok:
                api_key_prefix = (
                    (" (" + api_key[:2] + "*" * (len(api_key) - 4) + api_key[-2:] + ")") if len(api_key) > 4 else ""
                )
                raise ValueError(f"Invalid API key{api_key_prefix}: [{resp.status_code}] {resp.text}")
            info = resp.json()

            _check_org_info(info["org_info"], org_name)

            conn = _state.log_conn()
            conn.set_token(api_key)

        if not conn:
            raise ValueError(
                "Could not login to Braintrust. You may need to set BRAINTRUST_API_KEY in your environment."
            )

        # make_long_lived() allows the connection to retry if it breaks, which we're okay with after
        # this point because we know the connection _can_ successfully ping.
        conn.make_long_lived()

        # Set the same token in the API
        _state.api_conn().set_token(conn.token)
        _state.login_token = conn.token
        _state.logged_in = True


def log(**event):
    """
    Log a single event to the current experiment. The event will be batched and uploaded behind the scenes.

    :param **event: Data to be logged. See `Experiment.log` for full details.
    :returns: The `id` of the logged event.
    """

    current_experiment = _state.current_experiment.get()

    if not current_experiment:
        raise Exception("Not initialized. Please call init() first")

    return current_experiment.log(**event)


def summarize(summarize_scores=True, comparison_experiment_id=None):
    """
    Summarize the current experiment, including the scores (compared to the closest reference experiment) and metadata.

    :param summarize_scores: Whether to summarize the scores. If False, only the metadata will be returned.
    :param comparison_experiment_id: The experiment to compare against. If None, the most recent experiment on the comparison_commit will be used.
    :returns: `ExperimentSummary`
    """
    current_experiment = _state.current_experiment.get()

    if not current_experiment:
        raise Exception("Not initialized. Please call init() first")

    return current_experiment.summarize(
        summarize_scores=summarize_scores,
        comparison_experiment_id=comparison_experiment_id,
    )


def current_experiment() -> Optional["Experiment"]:
    """Returns the currently-active experiment (set by `with braintrust.init(...)` or `with braintrust.with_current(experiment)`). Returns undefined if no current experiment has been set."""

    return _state.current_experiment.get()


def current_logger() -> Optional["Logger"]:
    """Returns the currently-active logger (set by `with braintrust.init_logger(...)` or `with braintrust.with_current(logger)`). Returns undefined if no current experiment has been set."""

    return _state.current_logger.get()


def current_span() -> Span:
    """Return the currently-active span for logging (set by `with *.start_span` or `braintrust.with_current`). If there is no active span, returns a no-op span object, which supports the same interface as spans but does no logging.

    See `Span` for full details.
    """

    return _state.current_span.get()


def start_span(name=None, span_attributes={}, start_time=None, set_current=None, **event) -> Span:
    """Toplevel function for starting a span. It checks the following (in precedence order):
    * Currently-active span
    * Currently-active experiment
    * Currently-active logger

    and creates a span in the first one that is active. If none of these are active, it returns a no-op span object.

    Unless a name is explicitly provided, the name of the span will be the name of the calling function, or "root" if no meaningful name can be determined.

    We recommend running spans bound to a context manager (`with start_span`) to automatically mark them as current and ensure they are terminated. If you wish to start a span outside a callback, be sure to terminate it with `span.end()`.

    See `Span.startSpan` for full details.
    """

    name = name or get_caller_location()["caller_functionname"] or "root"
    kwargs = dict(name=name, span_attributes=span_attributes, start_time=start_time, set_current=set_current, **event)
    parent_span = current_span()
    if parent_span != NOOP_SPAN:
        return parent_span.start_span(**kwargs)

    experiment = current_experiment()
    if experiment:
        return experiment.start_span(**kwargs)

    logger = current_logger()
    if logger:
        return logger.start_span(**kwargs)

    return NOOP_SPAN


class _CurrentObjectWrapper:
    """Context manager wrapper for marking an experiment as current."""

    def __init__(self, object_cvar, object):
        self.object_cvar = object_cvar
        self.object = object

    def __enter__(self):
        self.context_token = self.object_cvar.set(self.object)

    def __exit__(self, type, value, callback):
        del type, value, callback

        self.object_cvar.reset(self.context_token)


def with_current(object: Union["Experiment", "Logger", "SpanImpl", _NoopSpan]):
    """Set the given experiment or span as current within the bound context manager (`with braintrust.with_current(object)`) and any asynchronous operations created within the block. The current experiment can be accessed with `braintrust.current_experiment`, and the current span with `braintrust.current_span`.

    :param object: The experiment or span to be marked as current.
    """
    if type(object) == Experiment:
        return _CurrentObjectWrapper(_state.current_experiment, object)
    elif type(object) == Logger:
        return _CurrentObjectWrapper(_state.current_logger, object)
    elif type(object) == SpanImpl or type(object) == _NoopSpan:
        return _CurrentObjectWrapper(_state.current_span, object)
    else:
        raise RuntimeError(f"Invalid object of type {type(object)}")


def traced(*span_args, **span_kwargs):
    """Decorator to trace the wrapped function as a span. Can either be applied bare (`@traced`) or by providing arguments (`@traced(*span_args, **span_kwargs)`), which will be forwarded to the created span. See `braintrust.start_span` for details on how the span is created, and `Span.start_span` for full details on the span arguments.

    Unless a name is explicitly provided in `span_args` or `span_kwargs`, the name of the span will be the name of the decorated function.
    """

    def decorator(span_args, span_kwargs, f):
        # We assume 'name' is the first positional argument in `start_span`.
        if len(span_args) == 0 and span_kwargs.get("name") is None:
            span_args += (f.__name__,)

        @wraps(f)
        def wrapper_sync(*f_args, **f_kwargs):
            with start_span(*span_args, **span_kwargs):
                return f(*f_args, **f_kwargs)

        @wraps(f)
        async def wrapper_async(*f_args, **f_kwargs):
            with start_span(*span_args, **span_kwargs):
                return await f(*f_args, **f_kwargs)

        if inspect.iscoroutinefunction(f):
            return wrapper_async
        else:
            return wrapper_sync

    # We determine if the decorator is invoked bare or with arguments by
    # checking if the first positional argument to the decorator is a callable.
    if len(span_args) == 1 and len(span_kwargs) == 0 and callable(span_args[0]):
        return decorator(span_args[1:], span_kwargs, span_args[0])
    else:
        return partial(decorator, span_args, span_kwargs)


def _check_org_info(org_info, org_name):
    global _state

    if len(org_info) == 0:
        raise ValueError("This user is not part of any organizations.")

    for orgs in org_info:
        if org_name is None or orgs["name"] == org_name:
            _state.org_id = orgs["id"]
            _state.org_name = orgs["name"]
            _state.log_url = os.environ.get("BRAINTRUST_LOG_URL", orgs["api_url"])
            break

    if _state.org_id is None:
        raise ValueError(
            f"Organization {org_name} not found. Must be one of {', '.join([x['name'] for x in org_info])}"
        )


def _save_api_info(api_info):
    os.makedirs(CACHE_PATH, exist_ok=True)
    with open(LOGIN_INFO_PATH, "w") as f:
        json.dump(api_info, f)


def _urljoin(*parts):
    return "/".join([x.lstrip("/") for x in parts])


def _populate_args(d, **kwargs):
    for k, v in kwargs.items():
        if v is not None:
            d[k] = v

    return d


def _validate_and_sanitize_experiment_log_partial_args(event):
    # Make sure only certain keys are specified.
    forbidden_keys = set(event.keys()) - {
        "input",
        "output",
        "expected",
        "scores",
        "metadata",
        "metrics",
        "dataset_record_id",
        "inputs",
    }
    if forbidden_keys:
        raise ValueError(f"The following keys may are not permitted: {forbidden_keys}")

    scores = event.get("scores")
    if scores:
        for name, score in scores.items():
            if not isinstance(name, str):
                raise ValueError("score names must be strings")

            if isinstance(score, bool):
                score = 1 if score else 0
                scores[name] = score

            if not isinstance(score, (int, float)):
                raise ValueError("score values must be numbers")
            if score < 0 or score > 1:
                raise ValueError("score values must be between 0 and 1")

    metadata = event.get("metadata")
    if metadata:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        for key in metadata.keys():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")

    metrics = event.get("metrics")
    if metrics:
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary")
        for key in metrics.keys():
            if not isinstance(key, str):
                raise ValueError("metric keys must be strings")

    input = event.get("input")
    inputs = event.get("inputs")
    if input is not None and inputs is not None:
        raise ValueError("Only one of input or inputs (deprecated) can be specified. Prefer input.")
    if inputs is not None:
        return dict(**{k: v for k, v in event.items() if k not in ["input", "inputs"]}, input=inputs)
    else:
        return {k: v for k, v in event.items()}


# Note that this only checks properties that are expected of a complete event.
# _validate_and_sanitize_experiment_log_partial_args should still be invoked
# (after handling special fields like 'id').
def _validate_and_sanitize_experiment_log_full_args(event, has_dataset):
    input = event.get("input")
    inputs = event.get("inputs")
    if (input is not None and inputs is not None) or (input is None and inputs is None):
        raise ValueError("Exactly one of input or inputs (deprecated) must be specified. Prefer input.")

    if event.get("scores") is None:
        raise ValueError("scores must be specified")
    elif not isinstance(event["scores"], dict):
        raise ValueError("scores must be a dictionary of names with scores")

    if has_dataset and event.get("dataset_record_id") is None:
        raise ValueError("dataset_record_id must be specified when using a dataset")
    elif not has_dataset and event.get("dataset_record_id") is not None:
        raise ValueError("dataset_record_id cannot be specified when not using a dataset")

    return event


class Experiment(ModelWrapper):
    """
    An experiment is a collection of logged events, such as model inputs and outputs, which represent
    a snapshot of your application at a particular point in time. An experiment is meant to capture more
    than just the model you use, and includes the data you use to test, pre- and post- processing code,
    comparison metrics (scores), and any other metadata you want to include.

    Experiments are associated with a project, and two experiments are meant to be easily comparable via
    their `inputs`. You can change the attributes of the experiments in a project (e.g. scoring functions)
    over time, simply by changing what you log.

    You should not create `Experiment` objects directly. Instead, use the `braintrust.init()` method.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str = None,
        description: str = None,
        dataset: "Dataset" = None,
        update: bool = False,
        base_experiment: str = None,
        is_public: bool = False,
        set_current: bool = None,
    ):
        self.finished = False
        self.set_current = True if set_current is None else set_current

        args = {"project_name": project_name, "org_id": _state.org_id}

        if experiment_name is not None:
            args["experiment_name"] = experiment_name

        if description is not None:
            args["description"] = description

        if update:
            args["update"] = update

        repo_status = get_repo_status()
        if repo_status:
            args["repo_info"] = repo_status.as_dict()

        if base_experiment is not None:
            args["base_experiment"] = base_experiment
        else:
            args["ancestor_commits"] = list(get_past_n_ancestors())

        if dataset is not None:
            args["dataset_id"] = dataset.id
            args["dataset_version"] = dataset.version

        if is_public is not None:
            args["public"] = is_public

        while True:
            try:
                response = _state.api_conn().post_json("api/experiment/register", args)
                break
            except AugmentedHTTPError as e:
                if args.get("base_experiment") is not None and "base experiment" in str(e):
                    _logger.warning(f"Base experiment {args['base_experiment']} not found.")
                    args["base_experiment"] = None
                else:
                    raise

        self.project = ModelWrapper(response["project"])
        super().__init__(response["experiment"])
        self.dataset = dataset
        self.logger = _LogThread(name=experiment_name)
        self.last_start_time = time.time()

        _unterminated_objects.add_unterminated(self, get_caller_location())

    def log(
        self,
        input=None,
        output=None,
        expected=None,
        scores=None,
        metadata=None,
        metrics=None,
        id=None,
        dataset_record_id=None,
        inputs=None,
    ):
        """
        Log a single event to the experiment. The event will be batched and uploaded behind the scenes.

        :param input: The arguments that uniquely define a test case (an arbitrary, JSON serializable object). Later on, Braintrust will use the `input` to know whether two test cases are the same between experiments, so they should not contain experiment-specific state. A simple rule of thumb is that if you run the same experiment twice, the `input` should be identical.
        :param output: The output of your application, including post-processing (an arbitrary, JSON serializable object), that allows you to determine whether the result is correct or not. For example, in an app that generates SQL queries, the `output` should be the _result_ of the SQL query generated by the model, not the query itself, because there may be multiple valid queries that answer a single question.
        :param expected: The ground truth value (an arbitrary, JSON serializable object) that you'd compare to `output` to determine if your `output` value is correct or not. Braintrust currently does not compare `output` to `expected` for you, since there are so many different ways to do that correctly. Instead, these values are just used to help you navigate your experiments while digging into analyses. However, we may later use these values to re-score outputs or fine-tune your models.
        :param scores: A dictionary of numeric values (between 0 and 1) to log. The scores should give you a variety of signals that help you determine how accurate the outputs are compared to what you expect and diagnose failures. For example, a summarization app might have one score that tells you how accurate the summary is, and another that measures the word similarity between the generated and grouth truth summary. The word similarity score could help you determine whether the summarization was covering similar concepts or not. You can use these scores to help you sort, filter, and compare experiments.
        :param metadata: (Optional) a dictionary with additional data about the test example, model outputs, or just about anything else that's relevant, that you can use to help find and analyze examples later. For example, you could log the `prompt`, example's `id`, or anything else that would be useful to slice/dice later. The values in `metadata` can be any JSON-serializable type, but its keys must be strings.
        :param metrics: (Optional) a dictionary of metrics to log. The following keys are populated automatically: "start", "end", "caller_functionname", "caller_filename", "caller_lineno".
        :param id: (Optional) a unique identifier for the event. If you don't provide one, BrainTrust will generate one for you.
        :param dataset_record_id: (Optional) the id of the dataset record that this event is associated with. This field is required if and only if the experiment is associated with a dataset.
        :param inputs: (Deprecated) the same as `input` (will be removed in a future version).
        :returns: The `id` of the logged event.
        """
        self._check_not_finished()

        event = _validate_and_sanitize_experiment_log_full_args(
            dict(
                input=input,
                output=output,
                expected=expected,
                scores=scores,
                metadata=metadata,
                metrics=metrics,
                id=id,
                dataset_record_id=dataset_record_id,
                inputs=inputs,
            ),
            self.dataset is not None,
        )
        span = self.start_span(start_time=self.last_start_time, **event)
        self.last_start_time = span.end()
        return span.id

    def start_span(self, name="root", span_attributes={}, start_time=None, set_current=None, **event):
        """Create a new toplevel span. The name parameter is optional and defaults to "root".

        See `Span.start_span` for full details
        """
        self._check_not_finished()

        return SpanImpl(
            bg_logger=self.logger,
            name=name,
            span_attributes=span_attributes,
            start_time=start_time,
            set_current=set_current,
            event=event,
            root_experiment=self,
        )

    def summarize(self, summarize_scores=True, comparison_experiment_id=None):
        """
        Summarize the experiment, including the scores (compared to the closest reference experiment) and metadata.

        :param summarize_scores: Whether to summarize the scores. If False, only the metadata will be returned.
        :param comparison_experiment_id: The experiment to compare against. If None, the most recent experiment on the origin's main branch will be used.
        :returns: `ExperimentSummary`
        """
        self._check_not_finished()

        # Flush our events to the API, and to the data warehouse, to ensure that the link we print
        # includes the new experiment.
        self.logger.flush()

        project_url = (
            f"{_state.api_url}/app/{encode_uri_component(_state.org_name)}/p/{encode_uri_component(self.project.name)}"
        )
        experiment_url = f"{project_url}/{encode_uri_component(self.name)}"

        score_summary = {}
        metric_summary = {}
        comparison_experiment_name = None
        if summarize_scores:
            # Get the comparison experiment
            if comparison_experiment_id is None:
                conn = _state.log_conn()
                resp = conn.get("/crud/base_experiments", params={"id": self.id})
                response_raise_for_status(resp)
                base_experiments = resp.json()
                if base_experiments:
                    comparison_experiment_id = base_experiments[0]["base_exp_id"]
                    comparison_experiment_name = base_experiments[0]["base_exp_name"]

            if comparison_experiment_id is not None:
                summary_items = _state.log_conn().get_json(
                    "experiment-comparison2",
                    args={
                        "experiment_id": self.id,
                        "base_experiment_id": comparison_experiment_id,
                    },
                    retries=3,
                )
                score_items = summary_items.get("scores", {})
                metric_items = summary_items.get("metrics", {})

                longest_score_name = max(len(k) for k in score_items.keys()) if score_items else 0
                score_summary = {
                    k: ScoreSummary(_longest_score_name=longest_score_name, **v) for (k, v) in score_items.items()
                }

                longest_metric_name = max(len(k) for k in metric_items.keys()) if metric_items else 0
                metric_summary = {
                    k: MetricSummary(_longest_metric_name=longest_metric_name, **v) for (k, v) in metric_items.items()
                }

        return ExperimentSummary(
            project_name=self.project.name,
            experiment_name=self.name,
            project_url=project_url,
            experiment_url=experiment_url,
            comparison_experiment_name=comparison_experiment_name,
            scores=score_summary,
            metrics=metric_summary,
        )

    def close(self):
        """Finish the experiment and return its id. After calling close, you may not invoke any further methods on the experiment object.

        Will be invoked automatically if the experiment is bound to a context manager.

        :returns: The experiment id.
        """
        self._check_not_finished()

        self.logger.flush()

        self.finished = True
        _unterminated_objects.remove_unterminated(self)
        return self.id

    def _check_not_finished(self):
        if self.finished:
            raise RuntimeError("Cannot invoke method on finished experiment")

    def __enter__(self):
        if self.set_current:
            self._context_token = _state.current_experiment.set(self)
        return self

    def __exit__(self, type, value, callback):
        del type, value, callback

        if self.set_current:
            _state.current_experiment.reset(self._context_token)

        self.close()


class SpanImpl(Span):
    """Primary implementation of the `Span` interface. See the `Span` interface for full details on each method.

    We suggest using one of the various `start_span` methods, instead of creating Spans directly. See `Span.start_span` for full details.
    """

    # root_experiment should only be specified for a root span. parent_span
    # should only be specified for non-root spans.
    def __init__(
        self,
        bg_logger,
        name,
        span_attributes={},
        start_time=None,
        set_current=None,
        event={},
        root_experiment=None,
        root_project=None,
        parent_span=None,
    ):
        if sum(x is not None for x in [root_experiment, root_project, parent_span]) != 1:
            raise ValueError("Must specify exactly one of `root_experiment`, `root_project`, and `parent_span`")

        self.finished = False
        self.set_current = True if set_current is None else set_current
        self._logged_end_time = None

        self.bg_logger = bg_logger

        # `internal_data` contains fields that are not part of the
        # "user-sanitized" set of fields which we want to log in just one of the
        # span rows.
        caller_location = get_caller_location()
        self.internal_data = dict(
            metrics=dict(
                start=start_time or time.time(),
                **(caller_location or {}),
            ),
            span_attributes=dict(**span_attributes, name=name),
            created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        id = event.get("id", None)
        if id is None:
            id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        # `_object_info` contains fields that are logged to every span row.
        if root_experiment is not None:
            self._object_info = dict(
                id=id,
                span_id=span_id,
                root_span_id=span_id,
                project_id=root_experiment.project.id,
                experiment_id=root_experiment.id,
            )
        elif root_project is not None:
            self._object_info = dict(
                id=id,
                span_id=span_id,
                root_span_id=span_id,
                org_id=_state.org_id,
                project_id=root_project.id,
                log_id="g",
            )
        elif parent_span is not None:
            self._object_info = {**parent_span._object_info}
            self._object_info.update(id=id, span_id=span_id)
            self.internal_data.update(span_parents=[parent_span.span_id])
        else:
            raise RuntimeError("Must provide either 'root_experiment', 'root_project', or 'parent_span'")

        # The first log is a replacement, but subsequent logs to the same span
        # object will be merges.
        self._is_merge = False
        self.log(**{k: v for k, v in event.items() if k != "id"})
        self._is_merge = True

        _unterminated_objects.add_unterminated(self, caller_location)

    @property
    def id(self):
        return self._object_info["id"]

    @property
    def span_id(self):
        return self._object_info["span_id"]

    @property
    def root_span_id(self):
        return self._object_info["root_span_id"]

    def log(self, **event):
        self._check_not_finished()

        sanitized = {
            k: v for k, v in _validate_and_sanitize_experiment_log_partial_args(event).items() if v is not None
        }
        # There should be no overlap between the dictionaries being merged,
        # except for `sanitized` and `internal_data`, where the former overrides
        # the latter.
        sanitized_and_internal_data = {**self.internal_data}
        merge_dicts(sanitized_and_internal_data, sanitized)
        record = dict(
            **sanitized_and_internal_data,
            **self._object_info,
            **{IS_MERGE_FIELD: self._is_merge},
        )
        if "metrics" in record and "end" in record["metrics"]:
            self._logged_end_time = record["metrics"]["end"]
        self.internal_data = {}
        self.bg_logger.log(record)

    def start_span(self, name, span_attributes={}, start_time=None, set_current=None, **event):
        self._check_not_finished()

        return SpanImpl(
            bg_logger=self.bg_logger,
            name=name,
            span_attributes=span_attributes,
            start_time=start_time,
            set_current=set_current,
            event=event,
            parent_span=self,
        )

    def end(self, end_time=None):
        self._check_not_finished()

        if not self._logged_end_time:
            end_time = end_time or time.time()
            self.internal_data = dict(metrics=dict(end=end_time))
        else:
            end_time = self._logged_end_time
        self.log()

        self.finished = True
        _unterminated_objects.remove_unterminated(self)
        return end_time

    def close(self, end_time=None):
        return self.end(end_time)

    def _check_not_finished(self):
        if self.finished:
            raise RuntimeError("Cannot invoke method on finished span")

    def __enter__(self):
        if self.set_current:
            self._context_token = _state.current_span.set(self)
        return self

    def __exit__(self, type, value, callback):
        del type, value, callback

        if self.set_current:
            _state.current_span.reset(self._context_token)

        self.end()


class Dataset(ModelWrapper):
    """
    A dataset is a collection of records, such as model inputs and outputs, which represent
    data you can use to evaluate and fine-tune models. You can log production data to datasets,
    curate them with interesting examples, edit/delete records, and run evaluations against them.

    You should not create `Dataset` objects directly. Instead, use the `braintrust.init_dataset()` method.
    """

    def __init__(self, project_name: str, name: str = None, description: str = None, version: "str | int" = None):
        self.finished = False

        args = _populate_args(
            {"project_name": project_name, "org_id": _state.org_id},
            dataset_name=name,
            description=description,
        )
        response = _state.api_conn().post_json("api/dataset/register", args)
        self.project = ModelWrapper(response["project"])

        self.new_records = 0

        self._fetched_data = None

        self._pinned_version = None
        if version is not None:
            try:
                self._pinned_version = int(version)
                assert self._pinned_version >= 0
            except (ValueError, AssertionError):
                raise ValueError(f"version ({version}) must be a positive integer")

        super().__init__(response["dataset"])
        self.logger = _LogThread(name=self.name)

        _unterminated_objects.add_unterminated(self, get_caller_location())

    def insert(self, input, output, metadata=None, id=None):
        """
        Insert a single record to the dataset. The record will be batched and uploaded behind the scenes. If you pass in an `id`,
        and a record with that `id` already exists, it will be overwritten (upsert).

        :param input: The argument that uniquely define an input case (an arbitrary, JSON serializable object).
        :param output: The output of your application, including post-processing (an arbitrary, JSON serializable object).
        :param metadata: (Optional) a dictionary with additional data about the test example, model outputs, or just
        about anything else that's relevant, that you can use to help find and analyze examples later. For example, you could log the
        `prompt`, example's `id`, or anything else that would be useful to slice/dice later. The values in `metadata` can be any
        JSON-serializable type, but its keys must be strings.
        :param id: (Optional) a unique identifier for the event. If you don't provide one, Braintrust will generate one for you.
        :returns: The `id` of the logged record.
        """
        self._check_not_finished()

        if metadata:
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            for key in metadata.keys():
                if not isinstance(key, str):
                    raise ValueError("metadata keys must be strings")

        args = _populate_args(
            {
                "id": id or str(uuid.uuid4()),
                "inputs": input,
                "output": output,
                "project_id": self.project.id,
                "dataset_id": self.id,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            metadata=metadata,
        )

        self._clear_cache()  # We may be able to optimize this
        self.new_records += 1
        self.logger.log(args)
        return args["id"]

    def delete(self, id):
        """
        Delete a record from the dataset.

        :param id: The `id` of the record to delete.
        """
        self._check_not_finished()

        args = _populate_args(
            {
                "id": id,
                "project_id": self.project.id,
                "dataset_id": self.id,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "_object_delete": True,  # XXX potentially place this in the logging endpoint
            },
        )

        self.logger.log(args)
        return args["id"]

    def summarize(self, summarize_data=True):
        """
        Summarize the dataset, including high level metrics about its size and other metadata.

        :param summarize_data: Whether to summarize the data. If False, only the metadata will be returned.
        :returns: `DatasetSummary`
        """
        self._check_not_finished()

        # Flush our events to the API, and to the data warehouse, to ensure that the link we print
        # includes the new experiment.
        self.logger.flush()

        project_url = (
            f"{_state.api_url}/app/{encode_uri_component(_state.org_name)}/p/{encode_uri_component(self.project.name)}"
        )
        dataset_url = f"{project_url}/d/{encode_uri_component(self.name)}"

        data_summary = None
        if summarize_data:
            data_summary_d = _state.log_conn().get_json(
                "dataset-summary",
                args={
                    "dataset_id": self.id,
                },
                retries=3,
            )
            data_summary = DataSummary(new_records=self.new_records, **data_summary_d)

        return DatasetSummary(
            project_name=self.project.name,
            dataset_name=self.name,
            project_url=project_url,
            dataset_url=dataset_url,
            data_summary=data_summary,
        )

    def fetch(self):
        """
        Fetch all records in the dataset.

        ```python
        for record in dataset.fetch():
            print(record)

        # You can also iterate over the dataset directly.
        for record in dataset:
            print(record)
        ```

        :returns: An iterator over the records in the dataset.
        """
        self._check_not_finished()

        for record in self.fetched_data:
            yield {
                "id": record.get("id"),
                "input": json.loads(record.get("input") or "null"),
                "output": json.loads(record.get("output") or "null"),
                "metadata": json.loads(record.get("metadata") or "null"),
            }

        self._clear_cache()

    def __iter__(self):
        self._check_not_finished()
        return self.fetch()

    @property
    def fetched_data(self):
        self._check_not_finished()
        if not self._fetched_data:
            resp = _state.log_conn().get(
                "object/dataset", params={"id": self.id, "fmt": "json", "version": self._pinned_version}
            )
            response_raise_for_status(resp)

            self._fetched_data = [json.loads(line) for line in resp.content.split(b"\n") if line.strip()]
        return self._fetched_data

    def _clear_cache(self):
        self._check_not_finished()
        self._fetched_data = None

    @property
    def version(self):
        self._check_not_finished()
        if self._pinned_version is not None:
            return self._pinned_version
        else:
            return max([int(record.get(TRANSACTION_ID_FIELD, 0)) for record in self.fetched_data] or [0])

    def close(self):
        """Terminate connection to the dataset and return its id. After calling close, you may not invoke any further methods on the dataset object.

        Will be invoked automatically if the dataset is bound as a context manager.

        :returns: The dataset id.
        """
        self._check_not_finished()

        self.logger.flush()

        self.finished = True
        _unterminated_objects.remove_unterminated(self)
        return self.id

    def _check_not_finished(self):
        if self.finished:
            raise RuntimeError("Cannot invoke method on finished dataset")

    def __enter__(self):
        return self

    def __exit__(self, type, value, callback):
        del type, value, callback
        self.close()


class Project:
    def __init__(self, name=None, id=None):
        self._name = name
        self._id = id
        self.init_lock = threading.RLock()

    def lazy_init(self):
        if self._id is None or self._name is None:
            with self.init_lock:
                if self._id is None:
                    response = _state.api_conn().post_json(
                        "api/project/register",
                        {
                            "project_name": self._name or GLOBAL_PROJECT,
                            "org_id": _state.org_id,
                        },
                    )
                    self._id = response["project"]["id"]
                    self._name = response["project"]["name"]
                elif self._name is None:
                    response = _state.api_conn().get_json("api/project", {"id": self._id})
                    self._name = response["name"]

        return self

    @property
    def id(self):
        self.lazy_init()
        return self._id

    @property
    def name(self):
        self.lazy_init()
        return self._name


class Logger:
    def __init__(self, lazy_login: Callable, project: Project, async_flush: bool = True, set_current: bool = None):
        self._lazy_login = lazy_login
        self._logged_in = False

        self.project = project
        self.async_flush = async_flush
        self.set_current = True if set_current is None else set_current

        self.logger = _LogThread()
        self.last_start_time = time.time()

    def log(
        self,
        input=None,
        output=None,
        expected=None,
        scores=None,
        metadata=None,
        metrics=None,
        id=None,
    ):
        """
        Log a single event. The event will be batched and uploaded behind the scenes.

        :param input: The arguments that uniquely define a user input(an arbitrary, JSON serializable object).
        :param output: The output of your application, including post-processing (an arbitrary, JSON serializable object), that allows you to determine whether the result is correct or not. For example, in an app that generates SQL queries, the `output` should be the _result_ of the SQL query generated by the model, not the query itself, because there may be multiple valid queries that answer a single question.
        :param expected: The ground truth value (an arbitrary, JSON serializable object) that you'd compare to `output` to determine if your `output` value is correct or not. Braintrust currently does not compare `output` to `expected` for you, since there are so many different ways to do that correctly. Instead, these values are just used to help you navigate while digging into analyses. However, we may later use these values to re-score outputs or fine-tune your models.
        :param scores: A dictionary of numeric values (between 0 and 1) to log. The scores should give you a variety of signals that help you determine how accurate the outputs are compared to what you expect and diagnose failures. For example, a summarization app might have one score that tells you how accurate the summary is, and another that measures the word similarity between the generated and grouth truth summary. The word similarity score could help you determine whether the summarization was covering similar concepts or not. You can use these scores to help you sort, filter, and compare logs.
        :param metadata: (Optional) a dictionary with additional data about the test example, model outputs, or just about anything else that's relevant, that you can use to help find and analyze examples later. For example, you could log the `prompt`, example's `id`, or anything else that would be useful to slice/dice later. The values in `metadata` can be any JSON-serializable type, but its keys must be strings.
        :param metrics: (Optional) a dictionary of metrics to log. The following keys are populated automatically: "start", "end", "caller_functionname", "caller_filename", "caller_lineno".
        :param id: (Optional) a unique identifier for the event. If you don't provide one, BrainTrust will generate one for you.
        """
        # Do the lazy login before retrieving the last_start_time.
        self._perform_lazy_login()
        span = self.start_span(
            start_time=self.last_start_time,
            input=input,
            output=output,
            expected=expected,
            scores=scores,
            metadata=metadata,
            metrics=metrics,
            id=id,
        )
        self.last_start_time = span.end()

        if not self.async_flush:
            self.logger.flush()

        return span.id

    def _perform_lazy_login(self):
        if not self._logged_in:
            self._lazy_login()
            self.last_start_time = time.time()
            self._logged_in = True

    def start_span(self, name="root", span_attributes={}, start_time=None, set_current=None, **event):
        """Create a new toplevel span. The name parameter is optional and defaults to "root".

        See `Span.start_span` for full details
        """
        self._perform_lazy_login()
        return SpanImpl(
            bg_logger=self.logger,
            name=name,
            span_attributes=span_attributes,
            start_time=start_time,
            set_current=set_current,
            event=event,
            root_project=self.project,
        )

    def __enter__(self):
        self._perform_lazy_login()
        if self.set_current:
            self._context_token = _state.current_logger.set(self)
        return self

    def __exit__(self, type, value, callback):
        del type, value, callback

        if self.set_current:
            _state.current_logger.reset(self._context_token)

        self.logger.flush()

    def flush(self):
        """
        Flush any pending logs to the server.
        """
        self._perform_lazy_login()
        self.logger.flush()


@dataclasses.dataclass
class ScoreSummary(SerializableDataClass):
    """Summary of a score's performance."""

    """Name of the score."""
    name: str
    """Average score across all examples."""
    score: float
    """Difference in score between the current and reference experiment."""
    diff: float
    """Number of improvements in the score."""
    improvements: int
    """Number of regressions in the score."""
    regressions: int

    # Used to help with formatting
    _longest_score_name: int

    def __str__(self):
        # format with 2 decimal points and pad so that it's exactly 2 characters then 2 decimals
        score_pct = f"{self.score * 100:05.2f}%"
        diff_pct = f"{abs(self.diff) * 100:05.2f}%"
        diff_score = f"+{diff_pct}" if self.diff > 0 else f"-{diff_pct}" if self.diff < 0 else "-"

        # pad the name with spaces so that its length is self._longest_score_name + 2
        score_name = f"'{self.name}'".ljust(self._longest_score_name + 2)

        return textwrap.dedent(
            f"""{score_pct} ({diff_score}) {score_name} score\t({self.improvements} improvements, {self.regressions} regressions)"""
        )


@dataclasses.dataclass
class MetricSummary(SerializableDataClass):
    """Summary of a metric's performance."""

    """Name of the metric."""
    name: str
    """Average metric across all examples."""
    metric: float
    """Unit label for the metric."""
    unit: str
    """Difference in metric between the current and reference experiment."""
    diff: float
    """Number of improvements in the metric."""
    improvements: int
    """Number of regressions in the metric."""
    regressions: int

    # Used to help with formatting
    _longest_metric_name: int

    def __str__(self):
        # format with 2 decimal points
        metric = f"{self.metric:.2f}"
        diff_pct = f"{abs(self.diff) * 100:05.2f}%"
        diff_score = f"+{diff_pct}" if self.diff > 0 else f"-{diff_pct}" if self.diff < 0 else "-"

        # pad the name with spaces so that its length is self._longest_score_name + 2
        metric_name = f"'{self.name}'".ljust(self._longest_metric_name + 2)

        return textwrap.dedent(
            f"""{metric}{self.unit} ({diff_score}) {metric_name}\t({self.improvements} improvements, {self.regressions} regressions)"""
        )


@dataclasses.dataclass
class ExperimentSummary(SerializableDataClass):
    """Summary of an experiment's scores and metadata."""

    """Name of the project that the experiment belongs to."""
    project_name: str
    """Name of the experiment."""
    experiment_name: str
    """URL to the project's page in the Braintrust app."""
    project_url: str
    """URL to the experiment's page in the Braintrust app."""
    experiment_url: str
    """The experiment scores are baselined against."""
    comparison_experiment_name: Optional[str]
    """Summary of the experiment's scores."""
    scores: Dict[str, ScoreSummary]
    """Summary of the experiment's metrics."""
    metrics: Dict[str, ScoreSummary]

    def __str__(self):
        comparison_line = ""
        if self.comparison_experiment_name:
            comparison_line = f"""{self.experiment_name} compared to {self.comparison_experiment_name}:\n"""
        return (
            f"""\n=========================SUMMARY=========================\n{comparison_line}"""
            + "\n".join([str(score) for score in self.scores.values()])
            + ("\n\n" if self.scores else "")
            + "\n".join([str(metric) for metric in self.metrics.values()])
            + ("\n\n" if self.metrics else "")
            + textwrap.dedent(
                f"""\
        See results for {self.experiment_name} at {self.experiment_url}"""
            )
        )


@dataclasses.dataclass
class DataSummary(SerializableDataClass):
    """Summary of a dataset's data."""

    """New or updated records added in this session."""
    new_records: int
    """Total records in the dataset."""
    total_records: int

    def __str__(self):
        return textwrap.dedent(f"""Total records: {self.total_records} ({self.new_records} new or updated records)""")


@dataclasses.dataclass
class DatasetSummary(SerializableDataClass):
    """Summary of a dataset's scores and metadata."""

    """Name of the project that the dataset belongs to."""
    project_name: str
    """Name of the dataset."""
    dataset_name: str
    """URL to the project's page in the Braintrust app."""
    project_url: str
    """URL to the experiment's page in the Braintrust app."""
    dataset_url: str
    """Summary of the dataset's data."""
    data_summary: int

    def __str__(self):
        return textwrap.dedent(
            f"""\

             =========================SUMMARY=========================
             {str(self.data_summary)}
             See results for all datasets in {self.project_name} at {self.project_url}
             See results for {self.dataset_name} at {self.dataset_url}"""
        )
````

## File: py/src/braintrust/merge_row_batch.py
````python
from .util import IS_MERGE_FIELD, merge_dicts

DATA_OBJECT_KEYS = [
    "org_id",
    "project_id",
    "experiment_id",
    "dataset_id",
    "prompt_session_id",
    "log_id",
]


def _generate_unique_row_key(row: dict):
    def coalesce_empty(field):
        return row.get(field, "")

    return ":".join([coalesce_empty(k) for k in DATA_OBJECT_KEYS + ["id"]])


def merge_row_batch(rows: list[dict]) -> list[dict]:
    out = []
    remaining_rows = []
    # First add any rows with no ID to `out`, since they will always be
    # independent.
    for row in rows:
        if row.get("id") is None:
            out.append(row)
        else:
            remaining_rows.append(row)
    row_groups = {}
    for row in remaining_rows:
        key = _generate_unique_row_key(row)
        existing_row = row_groups.get(key)
        if existing_row is not None and row.get(IS_MERGE_FIELD):
            preserve_nomerge = not existing_row.get(IS_MERGE_FIELD)
            merge_dicts(existing_row, row)
            if preserve_nomerge:
                del existing_row[IS_MERGE_FIELD]
        else:
            row_groups[key] = row
    out.extend(row_groups.values())
    return out
````

## File: py/src/braintrust/oai.py
````python
import time

from .logger import current_span


class NamedWrapper:
    def __init__(self, wrapped):
        self.__wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.__wrapped, name)


class ChatCompletionWrapper:
    def __init__(self, create_fn, acreate_fn):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn

    def create(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = current_span().start_span(name="OpenAI Chat Completion", **params)
        should_end = True
        try:
            start = time.time()
            raw_response = self.create_fn(*args, **kwargs)
            if stream:

                def gen():
                    try:
                        first = True
                        all_results = []
                        for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(item if isinstance(item, dict) else item.dict())
                            yield item
                        span.log(output=all_results)
                    finally:
                        span.end()

                should_end = False
                return gen()
            else:
                log_response = raw_response if isinstance(raw_response, dict) else raw_response.dict()
                span.log(
                    metrics={
                        "tokens": log_response["usage"]["total_tokens"],
                        "prompt_tokens": log_response["usage"]["prompt_tokens"],
                        "completion_tokens": log_response["usage"]["completion_tokens"],
                    },
                    output=log_response["choices"],
                )
                return raw_response
        finally:
            if should_end:
                span.end()

    async def acreate(self, *args, **kwargs):
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = current_span().start_span(name="OpenAI Chat Completion", **params)
        should_end = True
        try:
            start = time.time()
            raw_response = await self.acreate_fn(*args, **kwargs)
            if stream:

                async def gen():
                    try:
                        first = True
                        all_results = []
                        async for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(item if isinstance(item, dict) else item.dict())
                            yield item
                        span.log(output=all_results)
                    finally:
                        span.end()

                should_end = False
                return gen()
            else:
                log_response = raw_response if isinstance(raw_response, dict) else raw_response.dict()
                span.log(
                    metrics={
                        "tokens": log_response["usage"]["total_tokens"],
                        "prompt_tokens": log_response["usage"]["prompt_tokens"],
                        "completion_tokens": log_response["usage"]["completion_tokens"],
                    },
                    output=log_response["choices"],
                )
                return raw_response
        finally:
            if should_end:
                span.end()

    @classmethod
    def _parse_params(cls, params):
        params = {**params}
        messages = params.pop("messages", None)
        return {
            "input": messages,
            "metadata": params,
        }


class ChatCompletionV0Wrapper(NamedWrapper):
    def __init__(self, chat):
        self.__chat = chat
        super().__init__(chat)

    def create(self, *args, **kwargs):
        return ChatCompletionWrapper(self.__chat.create, self.__chat.acreate).create(*args, **kwargs)

    async def acreate(self, *args, **kwargs):
        return await ChatCompletionWrapper(self.__chat.create, self.__chat.acreate).acreate(*args, **kwargs)


# This wraps 0.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v0.28.1
class OpenAIV0Wrapper(NamedWrapper):
    def __init__(self, openai):
        super().__init__(openai)
        self.ChatCompletion = ChatCompletionV0Wrapper(openai.ChatCompletion)


class CompletionsV1Wrapper(NamedWrapper):
    def __init__(self, completions):
        self.__completions = completions
        super().__init__(completions)

    def create(self, *args, **kwargs):
        return ChatCompletionWrapper(self.__completions.create, None).create(*args, **kwargs)


class AsyncCompletionsV1Wrapper(NamedWrapper):
    def __init__(self, completions):
        self.__completions = completions
        super().__init__(completions)

    async def create(self, *args, **kwargs):
        return await ChatCompletionWrapper(None, self.__completions.create).acreate(*args, **kwargs)


class ChatV1Wrapper(NamedWrapper):
    def __init__(self, chat):
        super().__init__(chat)

        import openai

        if type(chat.completions) == openai.resources.chat.completions.AsyncCompletions:
            self.completions = AsyncCompletionsV1Wrapper(chat.completions)
        else:
            self.completions = CompletionsV1Wrapper(chat.completions)


# This wraps 1.*.* versions of the openai module, eg https://github.com/openai/openai-python/tree/v1.1.0
class OpenAIV1Wrapper(NamedWrapper):
    def __init__(self, openai):
        super().__init__(openai)
        self.chat = ChatV1Wrapper(openai.chat)


def wrap_openai(openai):
    """
    Wrap the openai module (pre v1) or OpenAI instance (post v1) to add tracing.
    If Braintrust is not configured, this is a no-op.

    :param openai: The openai module or OpenAI object
    """
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return OpenAIV1Wrapper(openai)
    else:
        return OpenAIV0Wrapper(openai)
````

## File: py/src/braintrust/resource_manager.py
````python
from contextlib import contextmanager
from threading import RLock


class ResourceManager:
    """A ResourceManager is a simple class to hold onto a shared resource. Local
    chalice is not thread-safe, so accessing shared memory across threads is not
    necessarily safe. But production AWS lambda will guarantee that memory is
    not shared across threads, so this synchronization is unnecessary.

    The ResourceManager controls access to a shared resource, optionally
    applying synchronization when run locally.
    """

    def __init__(self, resource):
        self.lock = RLock()
        self.resource = resource

    @contextmanager
    def get(self):
        with self.lock:
            yield self.resource
````

## File: py/src/braintrust/util.py
````python
import dataclasses
import inspect
import json
import os.path
import urllib.parse

from requests import HTTPError

GLOBAL_PROJECT = "Global"
TRANSACTION_ID_FIELD = "_xact_id"
IS_MERGE_FIELD = "_is_merge"


class SerializableDataClass:
    def as_dict(self):
        """Serialize the object to a dictionary."""
        return dataclasses.asdict(self)

    def as_json(self, **kwargs):
        """Serialize the object to JSON."""
        return json.dumps(self.as_dict(), **kwargs)


def encode_uri_component(name):
    """Encode a single component of a URI. Slashes are encoded as well, so this
    should not be used for multiple slash-separated URI components."""

    return urllib.parse.quote(name, safe="")


class AugmentedHTTPError(Exception):
    pass


def response_raise_for_status(resp):
    try:
        resp.raise_for_status()
    except HTTPError as e:
        raise AugmentedHTTPError(f"{resp.text}") from e


def get_caller_location():
    # Modified from
    # https://stackoverflow.com/questions/24438976/debugging-get-filename-and-line-number-from-which-a-function-is-called
    # to fetch the first stack frame not contained inside the same directory as
    # this file.
    this_dir = None
    call_stack = inspect.stack()
    for frame in call_stack:
        caller = inspect.getframeinfo(frame.frame)
        if this_dir is None:
            this_dir = os.path.dirname(caller.filename)
        if os.path.dirname(caller.filename) != this_dir:
            return dict(
                caller_functionname=caller.function,
                caller_filename=caller.filename,
                caller_lineno=caller.lineno,
            )
    return None


def merge_dicts(merge_into: dict, merge_from: dict):
    """Merges merge_from into merge_into, destructively updating merge_into."""

    if not isinstance(merge_into, dict):
        raise ValueError("merge_into must be a dictionary")
    if not isinstance(merge_from, dict):
        raise ValueError("merge_from must be a dictionary")

    for k, merge_from_v in merge_from.items():
        merge_into_v = merge_into.get(k)
        if isinstance(merge_into_v, dict) and isinstance(merge_from_v, dict):
            merge_dicts(merge_into_v, merge_from_v)
        else:
            merge_into[k] = merge_from_v
````

## File: py/src/braintrust/version.py
````python
VERSION = "0.0.74"
````

## File: py/.gitignore
````
src/braintrust.egg-info/
dist
````

## File: py/README.md
````markdown
## Braintrust

Tools for working with ML models.
````

## File: py/setup.py
````python
import os

import setuptools

dir_name = os.path.abspath(os.path.dirname(__file__))

version_contents = {}
with open(os.path.join(dir_name, "src", "braintrust", "version.py"), encoding="utf-8") as f:
    exec(f.read(), version_contents)

with open(os.path.join(dir_name, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = ["GitPython", "requests", "autoevals>=0.0.26", "tqdm"]

extras_require = {
    "cli": ["boto3", "psycopg2-binary"],
    "dev": [
        "black",
        "build",
        "flake8",
        "flake8-isort",
        "IPython",
        "isort==5.10.1",
        "pre-commit",
        "pytest",
        "twine",
    ],
    "doc": ["pydoc-markdown"],
}

extras_require["all"] = sorted({package for packages in extras_require.values() for package in packages})

setuptools.setup(
    name="braintrust",
    version=version_contents["VERSION"],
    author="Braintrust",
    author_email="info@braintrustdata.com",
    description="SDK for integrating Braintrust",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.braintrustdata.com",
    # project_urls={
    #    "Bug Tracker": "https://github.com/TODO/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7.0",
    entry_points={"console_scripts": ["braintrust = braintrust.cli.__main__:main"]},
    install_requires=install_requires,
    extras_require=extras_require,
)
````
