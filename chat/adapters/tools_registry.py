import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..domain import serialize_tool_result


class SalePriceArgs(BaseModel):
    original_price: float = Field(..., description="Original sticker price.")
    discount_percent: float = Field(..., description="Discount percentage.")


class ShippingEtaArgs(BaseModel):
    destination: str = Field(..., description="Destination city.")
    shipping_method: str = Field(default="standard", description="standard or express.")


@dataclass
class ToolExecutionResult:
    tool_id: str
    tool_name: str
    result: Any | None = None
    error: str | None = None
    llm_context: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    run: Callable[[Any], Awaitable[Any] | Any]
    format_result_for_llm: Callable[[Any], str] | None = None


async def calculate_sale_price(args: SalePriceArgs) -> dict[str, Any]:
    discounted_amount = round(args.original_price * args.discount_percent / 100, 2)
    final_price = round(args.original_price - discounted_amount, 2)
    return {
        "original_price": args.original_price,
        "discount_percent": args.discount_percent,
        "discount_amount": discounted_amount,
        "final_price": final_price,
        "currency": "USD",
    }


import time

def lookup_shipping_eta(args: ShippingEtaArgs) -> dict[str, Any]:
    time.sleep(1)  # 模擬同步阻塞式的網路請求
    return {
        "destination": args.destination,
        "shipping_method": args.shipping_method,
        "estimated_delivery": "1-2 business days",
    }


def build_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculate_sale_price",
            description="Calculate sale price.",
            args_schema=SalePriceArgs,
            run=calculate_sale_price,
            format_result_for_llm=lambda res: f"Result:\n{json.dumps(res, indent=2)}",
        ),
        ToolSpec(
            name="lookup_shipping_eta",
            description="Look up shipping ETA.",
            args_schema=ShippingEtaArgs,
            run=lookup_shipping_eta,
            format_result_for_llm=lambda res: f"Result:\n{json.dumps(res, indent=2)}",
        ),
    ]


def _make_langchain_tool(spec: ToolSpec) -> StructuredTool:
    async def _placeholder(**_: Any) -> str:
        raise RuntimeError("Fake func")

    return StructuredTool.from_function(
        coroutine=_placeholder,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_schema,
    )


async def _execute_tool_call(
    tool_call: dict[str, Any], tool_catalog: dict[str, ToolSpec]
) -> ToolExecutionResult:
    tool_id = tool_call.get("id", "")
    tool_name = tool_call.get("name", "")
    raw_args = tool_call.get("args", {})
    spec = tool_catalog.get(tool_name)

    if not spec:
        return ToolExecutionResult(
            tool_id=tool_id, tool_name=tool_name, error="Unknown tool"
        )
    try:
        validated = spec.args_schema.model_validate(raw_args)
        if inspect.iscoroutinefunction(spec.run):
            result = await spec.run(validated)
        else:
            # 同步函數丟進背景 Thread 執行，避免阻塞主 Event Loop
            result = await asyncio.to_thread(spec.run, validated)
        llm_context = (
            spec.format_result_for_llm(result)
            if spec.format_result_for_llm
            else serialize_tool_result(result)
        )
        return ToolExecutionResult(
            tool_id=tool_id, tool_name=tool_name, result=result, llm_context=llm_context
        )
    except Exception as exc:
        return ToolExecutionResult(tool_id=tool_id, tool_name=tool_name, error=str(exc))


DEFAULT_SPECS = build_tool_specs()
TOOL_CATALOG = {spec.name: spec for spec in DEFAULT_SPECS}
LANGCHAIN_TOOLS = [_make_langchain_tool(spec) for spec in DEFAULT_SPECS]
