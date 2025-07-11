# Braintrust Write logs Docs

URL Source: https://www.braintrust.dev/docs/guides/logs/write

There are many ways to log things from high level simple to more complex and customized [spans](https://www.braintrust.dev/docs/guides/traces/customize) for more control.

The simplest way to log to Braintrust is to wrap the code you wish to log with `@traced` for Python. This works for any function input and output provided. To learn more about tracing see the [tracing guide](https://www.braintrust.dev/docs/guides/traces).

Most commonly, logs are used for LLM calls. Braintrust includes a `wrap_openai` wrapper to be used on your OpenAI instance for the OpenAI API that automatically logs your requests. We _do not_ monkey patch the libraries directly.

Braintrust will automatically capture and log information behind the scenes.  You can see in in a a web-based logging and monitoring interface.  The most important elements of that interface are;:

- Logs and Monitoring Sections
- Trace and Span View such as a call to a "classify_text" tool with details
- Input and Output Details
- Table of Logs


You can use other AI model providers with the OpenAI client through the [AI proxy](https://www.braintrust.dev/docs/guides/proxy). 

```python
from openai import OpenAI
 
client = OpenAI(
    base_url="https://api.braintrust.dev/v1/proxy",
    api_key=os.environ["OPENAI_API_KEY"],  
)
 
response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "What is a proxy?"}],
    seed=1,  # A seed activates the proxy's cache
)
print(response.choices[0].message.content))
```

### Logging with `invoke`

```python
import os
 
from braintrust import init_logger, traced, wrap_openai
from openai import OpenAI
 
logger = init_logger(project="My Project")
 
client = wrap_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
 
 
@traced
def some_llm_function(input):
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Classify the following text as a question or a statement.",
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        model="gpt-4o",
    )
 
 
def my_route_handler(req):
    with logger.start_span() as span:
        body = req.body
        result = some_llm_function(body)
        span.log(input=body, output=result)
        return {
            "result": result,
            "request_id": span.id,
        }
 
 
# Assumes that the request is a JSON object with the requestId generated
# by the previous POST request, along with additional parameters like
# score (should be 1 for thumbs up and 0 for thumbs down), comment, and userId.
def my_feedback_handler(req):
    logger.log_feedback(
        id=req.body.request_id,
        scores={
            "correctness": req.body.score,
        },
        comment=req.body.comment,
        metadata={
            "user_id": req.user.id,
        },
    )
```

Braintrust supports logging user feedback, which can take multiple forms:

*   A **score** for a specific span, e.g. the output of a request could be thumbs-up (corresponding to 1) or thumbs-down (corresponding to 0).
*   An **expected** value, which gets saved in the `expected` field of a span, alongside `input` and `output`. This is a great place to store corrections.
*   A **comment** free-form text field that can be used to provide additional context.
*   Additional **metadata** fields, which allow you to track information about the feedback, like the `user_id` or `session_id`.

Each time you submit feedback, you can specify one or more of these fields using the `logFeedback()` / `log_feedback()` method, which needs you to specify the `span_id` corresponding to the span you want to log feedback for, and the feedback fields you want to update. As you log user feedback, the fields will update in real time.

The following example shows how to log feedback within a simple API endpoint.

### Collecting multiple scores

Often, you want to collect multiple scores for a single span. For example, multiple users might provide independent feedback on a single document. Although each score and expected value is logged separately, each update overwrites the previous value. Instead, to capture multiple scores, you should create a new span for each submission, and log the score in the `scores` field. When you view and use the trace, Braintrust will automatically average the scores for you in the parent span(s).

```python
import os
 
from braintrust import init_logger, traced, wrap_openai
from openai import OpenAI
 
logger = init_logger(project="My Project")
 
client = wrap_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
 
 
@traced
def some_llm_function(input):
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Classify the following text as a question or a statement.",
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        model="gpt-4o",
    )
 
 
def my_route_handler(req):
    with logger.start_span() as span:
        body = req.body
        result = some_llm_function(body)
        span.log(input=body, output=result)
        return {
            "result": result,
            "request_id": span.export(),
        }
 
 
def my_feedback_handler(req):
    with logger.start_span("feedback", parent=req.body.request_id) as span:
        logger.log_feedback(
            id=span.id,  # Use the newly created span's id, instead of the original request's id
            scores={
                "correctness": req.body.score,
            },
            comment=req.body.comment,
            metadata={
                "user_id": req.user.id,
            },
        )
```

### Data model

*   Each log entry is associated with an organization and a project.
*   Log entries contain optional `input`, `output`, `expected`, `scores`, `metadata`, and `metrics` fields. These fields are mandatory

### Initializing

The `init_logger()` method initializes the logger. Unlike the experiment `init()` method, the logger lazily initializes itself, so that you can call `init_logger()` at the top of your file (in module scope). The first time you `log()` or start a span, the logger will log into Braintrust and retrieve/initialize project details.