"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket


def _load_env_key() -> Optional[str]:
    """Đọc GEMINI_API_KEY từ biến môi trường hoặc file .env (nếu có)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    for env_path in [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except OSError:
                pass
    return None

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — Trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & chăm sóc khách hàng cho hệ sinh thái Vingroup (VinFast, Vinpearl).
- Giọng điệu: Chuyên nghiệp, lịch sự, tận tâm, rõ ràng và chính xác.

## 2. AVAILABLE TOOLS
Bạn được cung cấp các công cụ sau để tra cứu dữ liệu thực tế và xử lý yêu cầu:
{tools}
- `search_product_catalog(category, max_price)`: Tra cứu sản phẩm xe điện (`xe_dien`) hoặc dịch vụ nghỉ dưỡng (`du_lich`) theo mức giá tối đa.
- `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận phiếu yêu cầu hỗ trợ kỹ thuật hoặc khiếu nại của khách hàng.

## 3. CORE RULES
1. KHÔNG BAO GIỜ tự bịa đặt (hallucinate) tên sản phẩm, giá bán, thông số kỹ thuật hoặc mã ticket hỗ trợ.
2. BẮT BUỘC gọi công cụ `search_product_catalog` khi khách hàng hỏi tìm kiếm, xem sản phẩm hoặc lọc theo giá.
3. BẮT BUỘC gọi công cụ `submit_support_ticket` khi khách hàng báo lỗi, phản ánh sự cố hoặc yêu cầu lập phiếu hỗ trợ. Khi trả lời sau khi tạo ticket, BẮT BUỘC nêu rõ tên khách hàng và mã ticket (`TK-...`).
4. Nếu kết quả tra cứu rỗng (`[]`), BẮT BUỘC trả lời có cụm từ "Rất tiếc, không tìm thấy sản phẩm phù hợp".
5. Đối với câu hỏi về chính sách bảo hành pin VinFast (FAQ), trả lời trực tiếp không cần gọi tool: Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km.

## 4. OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ các thông tin, sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes, Vinmec, VinUni).
- Từ chối lịch sự các câu hỏi ngoài phạm vi hoặc yêu cầu can thiệp trái phép vào hệ thống.

## 5. OUTPUT CONTRACT
Tuân thủ quy trình suy luận ReAct theo định dạng:
- Thought: Phân tích yêu cầu của khách hàng và xác định công cụ cần gọi.
- Action: Tên công cụ và tham số đầu vào (JSON).
- Observation: Kết quả trả về từ công cụ.
- Final Answer: Câu trả lời tổng hợp đầy đủ, rõ ràng dành cho khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = _load_env_key()

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        if self.api_key:
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model_name}:generateContent?key={self.api_key}"
                )
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": user_input}]}]
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                answer_text = "".join(p.get("text", "") for p in parts).strip()
                if answer_text:
                    return {
                        "answer": answer_text,
                        "tool_calls": [],
                        "status": "success",
                        "mode": "live_gemini_baseline",
                    }
            except Exception:
                # Tự động fallback về mock nếu mạng/API key lỗi
                pass

        text_lower = user_input.lower()
        if "xe" in text_lower or "vinfast" in text_lower:
            answer = (
                "[Chatbot Baseline - Hallucination Warning] VinFast hiện có mẫu VinFast VF Pro Max "
                "giá 450 triệu đồng, quãng đường 900 km/lần sạc (thông tin tự suy đoán không qua catalog)."
            )
        elif "lỗi" in text_lower or "hỗ trợ" in text_lower or "phản hồi" in text_lower:
            answer = (
                "[Chatbot Baseline - Hallucination Warning] Đã ghi nhận hỗ trợ với mã phiếu #TK-FAKE-999 "
                "(thông tin giả định, thực tế chưa lưu vào hệ thống)."
            )
        else:
            answer = f"[Chatbot Baseline] Trả lời tự do (không dùng tool) cho: {user_input}"

        return {
            "answer": answer,
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5, model_name: str = "gemini-2.5-flash"):
        self.max_iterations = max_iterations
        self.model_name = model_name
        self.api_key = _load_env_key()
        self.trace: List[Dict[str, Any]] = []

    def _call_gemini_api(self, contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gọi Gemini REST API kèm System Prompt và TOOL_DEFINITIONS."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        formatted_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        )
        payload = {
            "systemInstruction": {
                "parts": [{"text": formatted_prompt}]
            },
            "contents": contents,
            "tools": [{"functionDeclarations": TOOL_DEFINITIONS}],
            "generationConfig": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _run_with_gemini(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Chạy ReAct / Tool Calling Loop trực tiếp với Gemini API."""
        contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_input}]}
        ]
        iteration = 1
        tool_Steps = 0

        while iteration <= self.max_iterations:
            response_data = self._call_gemini_api(contents)
            candidate = response_data.get("candidates", [{}])[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            text_parts = [p.get("text", "") for p in parts if "text" in p]

            if not function_calls:
                final_answer = "\n".join(t for t in text_parts if t).strip()
                effective_iterations = max(1, tool_Steps)
                self.trace.append({
                    "iteration": effective_iterations,
                    "thought": "Tổng hợp câu trả lời cuối cùng từ Gemini.",
                    "action": "final_answer",
                    "observation": final_answer,
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": effective_iterations,
                    "status": "completed",
                    "mode": "live_gemini_agent",
                }

            contents.append(content)
            response_parts = []
            for fc in function_calls:
                tool_Steps += 1
                fn_name = fc.get("name", "")
                fn_args = fc.get("args", {})
                tool_fn = TOOL_MAP.get(fn_name)
                if tool_fn:
                    observation = tool_fn(**fn_args)
                else:
                    observation = {"error": f"Unknown tool: {fn_name}"}

                self.trace.append({
                    "iteration": tool_Steps,
                    "thought": f"Gemini quyết định gọi tool {fn_name}.",
                    "action": fn_name,
                    "action_input": fn_args,
                    "observation": observation,
                })
                response_parts.append({
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": observation},
                    }
                })

            contents.append({"role": "user", "parts": response_parts})
            iteration += 1

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa (max_iterations_reached).",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
            "mode": "live_gemini_agent",
        }

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent và trích xuất tham số từ câu hỏi của người dùng."""
        text_lower = user_input.lower()

        # 1. Kiểm tra intent tạo ticket hỗ trợ
        ticket_keywords = [
            "bị lỗi", "lỗi", "sự cố", "hỏng", "phản hồi", "khiếu nại",
            "xử lý gấp", "bị ẩm mốc", "vấn đề nghiêm trọng", "ghi nhận phản hồi",
            "hỗ trợ kỹ thuật", "ticket"
        ]
        needs_ticket = any(kw in text_lower for kw in ticket_keywords)

        # 2. Kiểm tra intent tra cứu catalog sản phẩm
        catalog_action_keywords = [
            "muốn xem", "cho tôi xem", "tìm", "mua", "giá dưới", "dưới ",
            "có xe điện nào", "có xe nào", "resort", "tư vấn mẫu"
        ]
        is_faq_policy = ("chính sách" in text_lower or "kéo dài bao lâu" in text_lower) and not any(
            kw in text_lower for kw in ["giá dưới", "muốn xem", "cho tôi xem"]
        )
        needs_catalog = any(kw in text_lower for kw in catalog_action_keywords) and not is_faq_policy

        # Trích xuất tham số cho search_product_catalog
        category = "xe_dien"
        if any(kw in text_lower for kw in ["resort", "du lịch", "nghỉ dưỡng", "khách sạn"]):
            if "xe điện" not in text_lower or "xem resort" in text_lower:
                category = "du_lich"

        max_price = 999999999999
        price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|nghìn|ngàn)", text_lower)
        if price_match:
            val = float(price_match.group(1).replace(",", "."))
            unit = price_match.group(2)
            if unit == "tỷ":
                max_price = int(val * 1_000_000_000)
            elif unit == "triệu":
                max_price = int(val * 1_000_000)
            else:
                max_price = int(val * 1_000)

        # Trích xuất tham số cho submit_support_ticket
        customer_name = "Khách hàng"
        name_match = re.search(
            r"(?:tôi tên là|tên tôi là|tôi tên|tên là)\s+([^,.;]+)",
            user_input,
            flags=re.IGNORECASE
        )
        if name_match:
            customer_name = name_match.group(1).strip()

        issue_description = user_input
        if name_match:
            after_name = user_input[name_match.end():].lstrip(" ,:-")
            clause_match = re.split(r"\.\s*|,?\s*mức độ|(?:\.\s*Đây là)", after_name, maxsplit=1, flags=re.IGNORECASE)
            if clause_match and clause_match[0].strip():
                issue_description = clause_match[0].strip()

        priority = "medium"
        if any(kw in text_lower for kw in ["nghiêm trọng", "xử lý gấp", "khẩn cấp", "gấp", "cao", "high"]):
            priority = "high"
        elif any(kw in text_lower for kw in ["trung bình", "medium", "vừa"]):
            priority = "medium"
        elif any(kw in text_lower for kw in ["thấp", "low", "nhỏ"]):
            priority = "low"

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": not needs_catalog and not needs_ticket,
            "catalog_args": {
                "category": category,
                "max_price": max_price
            },
            "ticket_args": {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }
        }

    def _format_catalog_answer(self, results: List[Dict[str, Any]], args: Dict[str, Any]) -> str:
        """Định dạng câu trả lời từ kết quả tra cứu catalog."""
        if not results:
            return (
                f"Rất tiếc, không tìm thấy sản phẩm phù hợp trong danh mục '{args['category']}' "
                f"với mức giá dưới {args['max_price']:,} VNĐ."
            )
        lines = ["Dưới đây là các sản phẩm/dịch vụ Vingroup phù hợp với yêu cầu của quý khách:"]
        for item in results:
            price_str = f"{item['price_vnd']:,} VNĐ"
            lines.append(f"- {item['name']} ({price_str}): {item['description']}")
        return "\n".join(lines)

    def _format_ticket_answer(self, ticket_res: Dict[str, Any]) -> str:
        """Định dạng câu trả lời từ kết quả tạo ticket."""
        return (
            f"Đã ghi nhận yêu cầu hỗ trợ cho khách hàng {ticket_res['customer_name']}. "
            f"Mã phiếu hỗ trợ của quý khách là {ticket_res['ticket_id']} "
            f"(Mức độ ưu tiên: {ticket_res['priority']}, Trạng thái: {ticket_res['status']})."
        )

    def _format_faq_answer(self, user_input: str) -> str:
        """Trả lời trực tiếp các câu hỏi chính sách/FAQ chung của Vingroup."""
        text_lower = user_input.lower()
        if "bảo hành" in text_lower or "pin" in text_lower:
            return (
                "Chính sách bảo hành xe điện và pin của VinFast kéo dài lên đến 10 năm "
                "hoặc 200.000 km (tùy điều kiện nào đến trước), đảm bảo sự an tâm tối đa cho khách hàng."
            )
        return (
            "Xin chào quý khách! Tôi là VinAssistant — trợ lý ảo hỗ trợ tra cứu sản phẩm "
            "VinFast, Vinpearl và tiếp nhận yêu cầu hỗ trợ khách hàng."
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop (hỗ trợ cả Live Gemini API và Mock Simulator)."""
        self.trace = []

        # Nếu có GEMINI_API_KEY (trong .env hoặc biến môi trường), gọi trực tiếp Gemini LLM
        if self.api_key:
            try:
                live_result = self._run_with_gemini(user_input)
                if live_result:
                    return live_result
            except Exception as e:
                print(f"[Warning] Gemini API call failed ({e}), falling back to mock simulator.")
                self.trace = []

        # TODO 3: Phân tích intent từ user_input
        intents = self._detect_intents(user_input)
        needs_catalog = intents["needs_catalog"]
        needs_ticket = intents["needs_ticket"]

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        iteration = 1
        catalog_results = None
        ticket_result = None

        while iteration <= self.max_iterations:
            # Trường hợp cần cả 2 tools (Multi-step / Parallel Tool Calling)
            if needs_catalog and needs_ticket:
                if iteration == 1:
                    args = intents["catalog_args"]
                    tool_fn = TOOL_MAP["search_product_catalog"]
                    catalog_results = tool_fn(**args)
                    self.trace.append({
                        "iteration": iteration,
                        "thought": "Khách hàng muốn tra cứu sản phẩm, cần gọi search_product_catalog trước.",
                        "action": "search_product_catalog",
                        "action_input": args,
                        "observation": catalog_results
                    })
                elif iteration == 2:
                    args = intents["ticket_args"]
                    tool_fn = TOOL_MAP["submit_support_ticket"]
                    ticket_result = tool_fn(**args)
                    self.trace.append({
                        "iteration": iteration,
                        "thought": "Tiếp theo cần ghi nhận phản hồi của khách hàng qua submit_support_ticket.",
                        "action": "submit_support_ticket",
                        "action_input": args,
                        "observation": ticket_result
                    })
                else:
                    final_answer = (
                        self._format_catalog_answer(catalog_results or [], intents["catalog_args"])
                        + "\n\n"
                        + self._format_ticket_answer(ticket_result or {})
                    )
                    self.trace.append({
                        "iteration": iteration,
                        "thought": "Đã thu thập đủ kết quả từ cả hai công cụ, tổng hợp Final Answer.",
                        "action": "final_answer",
                        "observation": final_answer
                    })
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }

            # Trường hợp chỉ cần tra cứu catalog (Single Tool: search_product_catalog)
            elif needs_catalog:
                args = intents["catalog_args"]
                tool_fn = TOOL_MAP["search_product_catalog"]
                catalog_results = tool_fn(**args)
                final_answer = self._format_catalog_answer(catalog_results, args)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Khách hàng cần tìm kiếm sản phẩm theo danh mục và mức giá.",
                    "action": "search_product_catalog",
                    "action_input": args,
                    "observation": catalog_results,
                    "final_answer": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Trường hợp chỉ cần tạo ticket hỗ trợ (Single Tool: submit_support_ticket)
            elif needs_ticket:
                args = intents["ticket_args"]
                tool_fn = TOOL_MAP["submit_support_ticket"]
                ticket_result = tool_fn(**args)
                final_answer = self._format_ticket_answer(ticket_result)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Khách hàng báo cáo sự cố cần lập phiếu hỗ trợ kỹ thuật.",
                    "action": "submit_support_ticket",
                    "action_input": args,
                    "observation": ticket_result,
                    "final_answer": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Trường hợp câu hỏi FAQ (không cần gọi tool)
            else:
                final_answer = self._format_faq_answer(user_input)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Câu hỏi về chính sách chung (FAQ), không cần gọi tool.",
                    "action": None,
                    "observation": None,
                    "final_answer": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            iteration += 1

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa (max_iterations_reached).",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    api_key_detected = bool(_load_env_key())
    print(f"=== CHẾ ĐỘ CHẠY: {'LIVE GEMINI API (Đã nhận GEMINI_API_KEY)' if api_key_detected else 'MOCK SIMULATOR (Chưa đặt GEMINI_API_KEY)'} ===\n")

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
