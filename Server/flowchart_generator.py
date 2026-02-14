import os
import json
import re
import langgraph.graph as lg
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY environment variable")

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
)

def fix_class_diagram_syntax(code: str) -> str:
    """
    Detect class declarations with members appended without braces (e.g. "class BankAccount+accountNumber: string")
    and wrap the member definitions in curly braces with proper formatting.
    """
    pattern = r'^(class\s+\w+)([+\-#].+)$'
    def repl(match):
        line = match.group(0)
        if "{" in line:
            return line
        return f"{match.group(1)} {{\n  {match.group(2)}\n}}"
    fixed_code = re.sub(pattern, repl, code, flags=re.MULTILINE)
    return fixed_code

# Fix functions for specific diagram types.
def fix_quadrant_chart(code: str) -> str:
    """
    Mermaid quadrant charts can be picky. Remove the space between 'quadrant' and the number.
    For example, change "quadrant 1:" to "quadrant1:".
    """
    code = re.sub(r'(quadrant)\s+(\d+):', r'\1\2:', code)
    return code

def fix_sankey(code: str) -> str:
    """
    Ensure arrow syntax in sankey diagrams is consistently spaced.
    """
    code = re.sub(r'\s*--\s*', ' -- ', code)
    code = re.sub(r'\s*-->\s*', ' --> ', code)
    return code

def fix_journey(code: str) -> str:
    """
    For journey diagrams, ensure that each step is formatted as "Step Text: number: Actor".
    If extra colons occur in the step description, rejoin them so that only two colons remain.
    """
    fixed_lines = []
    for line in code.splitlines():
        # Only process lines that appear to be journey steps (indentation + text with colons)
        if re.match(r'^\s+\S+', line) and line.count(":") > 2:
            parts = line.split(":")
            # Rejoin all parts except the last two for the step description.
            step = ":".join(parts[:-2]).strip()
            fixed_line = f"{step}: {parts[-2].strip()}: {parts[-1].strip()}"
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)

def fix_xy_chart(code: str) -> str:
    """
    Ensure axis label syntax is correct by removing extra spaces before the colon.
    """
    code = re.sub(r'(xAxis|yAxis)\s+label:', r'\1 label:', code)
    return code

def fix_requirement_diagram(code: str) -> str:
    """
    For requirement diagrams, force the opening brace to be on a new line with proper indentation.
    """
    code = re.sub(r'\{\s*', '{\n  ', code)
    code = re.sub(r'\s*\}', '\n}', code)
    return code

def post_process_code(code: str) -> str:
    """
    Apply post-processing fixes based on the diagram type.
    """
    if code.startswith("classDiagram"):
        code = fix_class_diagram_syntax(code)
    if code.startswith("quadrantChart"):
        code = fix_quadrant_chart(code)
    if code.startswith("sankey"):
        code = fix_sankey(code)
    if code.startswith("journey"):
        code = fix_journey(code)
    if code.startswith("xyChart"):
        code = fix_xy_chart(code)
    if code.startswith("requirementDiagram"):
        code = fix_requirement_diagram(code)
    return code

def process_query(state: dict) -> dict:
    system_prompt = (
        "{\"code\": \"mermaid_code\"}\n\nGenerate STRICT JSON with PROPER SYNTAX. The output must always follow this structure:\n\n"
        "{\n  \"code\": \"mermaid_code\"\n}\n\nBelow are instructions and examples for various Mermaid diagram types. Use ONLY double quotes and escape newlines with \\n. Do NOT use markdown formatting.\n\n"
        "1. If the User Provides Details (e.g., for a Class Diagram):\nExample Query:\n\"Create a detailed class diagram for a banking system with:\\n- Account base class with balance attribute\\n- SavingsAccount and CheckingAccount subclasses\\n- Transaction class with relationships\\n- Proper visibility modifiers and data types\"\n\n"
        "Expected Output:\n{\"code\":\"classDiagram\\n    class Account {\\n        - balance: double\\n        + deposit(amount: double)\\n        + withdraw(amount: double)\\n    }\\n\\n    class SavingsAccount {\\n        - interestRate: double\\n        + calculateInterest(): double\\n    }\\n\\n    class CheckingAccount {\\n        - overdraftLimit: double\\n    }\\n\\n    class Transaction {\\n        - type: string\\n        - amount: double\\n        - date: Date\\n        + execute()\\n    }\\n\\n    Account <|-- SavingsAccount\\n    Account <|-- CheckingAccount\\n    Account *-- Transaction : hasTransactions\"}\n\n"
        "2. If the User Provides No Specific Details (e.g., a simple Class Diagram):\nExample Query:\n\"Create a class diagram for a banking system\"\n\n"
        "Expected Output:\n{\"code\":\"classDiagram\\n    class Bank {\\n        + name: string\\n        + location: string\\n    }\\n\\n    class Customer {\\n        + name: string\\n        + accountNumber: string\\n    }\\n\\n    class Account {\\n        + balance: double\\n        + deposit(amount: double)\\n        + withdraw(amount: double)\\n    }\\n\\n    Bank *-- Customer : has\\n    Customer *-- Account : owns\"}\n\n"
        "3. Flowchart Diagrams:\nSimple Flowchart Example:\nExample Query:\n\"Create a flowchart for a banking process\"\n\n"
        "Expected Output:\n{\"code\":\"graph TD\\n    Start --> Transaction[Perform Transaction]\\n    Transaction --> End\"}\n\n"
        "Detailed Flowchart Example:\nExample Query:\n\"Create a detailed flowchart for a banking transaction\"\n\n"
        "Expected Output:\n{\"code\":\"graph TD\\n    Start --> |Select Transaction| Decision{Deposit or Withdraw}\\n    Decision -->|Deposit| Process[Process Deposit]\\n    Decision -->|Withdraw| CheckBalance{Enough Balance?}\\n    CheckBalance -->|Yes| ProcessWithdraw[Process Withdrawal]\\n    CheckBalance -->|No| Reject[Reject Transaction]\\n    Process --> UpdateBalance[Update Balance]\\n    ProcessWithdraw --> UpdateBalance\\n    UpdateBalance --> End\\n    Reject --> End\"}\n\n"
        "4. Sequence Diagrams:\nExample Query:\n\"Create a sequence diagram for a user login process\"\n\n"
        "Expected Output:\n{\"code\":\"sequenceDiagram\\n    participant User\\n    participant System\\n    User->>System: Login Request\\n    System-->>User: Login Success\"}\n\n"
        "5. ER Diagrams:\nExample Query:\n\"Create an ER diagram for an order management system\"\n\n"
        "Expected Output:\n{\"code\":\"erDiagram\\n    CUSTOMER ||--o{ ORDER : places\\n    ORDER ||--|{ PRODUCT : contains\"}\n\n"
        "6. Gantt Diagrams:\nExample Query:\n\"Create a Gantt diagram for a project timeline\"\n\n"
        "Expected Output:\n{\"code\":\"gantt\\n    dateFormat  YYYY-MM-DD\\n    title Project Timeline\\n    section Planning\\n    Task A :a1, 2023-01-01, 10d\\n    section Development\\n    Task B :after a1, 20d\"}\n\n"
        "7. Mindmap Diagrams:\nExample Query:\n\"Create a mindmap for brainstorming ideas\"\n\n"
        "Expected Output:\n{\"code\":\"mindmap\\n  root((Central Idea))\\n    branch1((Sub Idea 1))\\n    branch2((Sub Idea 2))\"}\n\n"
        "8. State Diagrams:\nExample Query:\n\"Create a state diagram for a ticket booking system\"\n\n"
        "Expected Output:\n{\"code\":\"stateDiagram-v2\\n    [*] --> Idle\\n    Idle --> Booking\\n    Booking --> Confirmed\\n    Confirmed --> [*]\"}\n\n"
        "9. Timeline Diagrams:\nExample Query:\n\"Create a timeline diagram for company milestones\"\n\n"
        "Expected Output:\n{\"code\":\"timeline\\n    title Company Milestones\\n    2023-01-01 : Founded\\n    2023-06-01 : First Product Launch\"}\n\n"
        "10. Git Diagrams:\nExample Query:\n\"Create a Git diagram for a feature branch workflow\"\n\n"
        "Expected Output:\n{\"code\":\"gitGraph\\n    commit\\n    branch feature\\n    commit\\n    checkout feature\\n    commit\\n    merge feature\"}\n\n"
        "11. C4 Diagrams:\nExample Query:\n\"Create a C4 context diagram for an e-commerce system\"\n\n"
        "Expected Output:\n{\"code\":\"C4Context\\n    Person(customer, \\\"Customer\\\", \\\"A customer\\\")\\n    System(system, \\\"E-Commerce Platform\\\", \\\"Handles orders and payments\\\")\\n    Rel(customer, system, \\\"Uses\\\")\"}\n\n"
        "12. Sankey Diagrams:\nExample Query:\n\"Create a Sankey diagram for energy flow\"\n\n"
        "Expected Output:\n{\"code\":\"sankey\\n    A[Energy Source] -- 100 --> B[Conversion]\\n    B -- 60 --> C[Useful Energy]\\n    B -- 40 --> D[Losses]\"}\n\n"
        "13. Block Diagrams:\nExample Query:\n\"Create a block diagram for a simple system architecture\"\n\n"
        "Expected Output:\n{\"code\":\"flowchart LR\\n    A[Component A] --> B[Component B]\\n    B --> C[Component C]\"}\n\n"
        "14. Pie Charts:\nExample Query:\n\"Create a pie chart for market share distribution\"\n\n"
        "Expected Output:\n{\"code\":\"pie\\n    title Market Share\\n    \\\"Product A\\\" : 40\\n    \\\"Product B\\\" : 35\\n    \\\"Product C\\\" : 25\"}\n\n"
        "15. Quadrant Diagrams:\nExpected Input: Provide a title and labels for each quadrant.\nExample Query:\n\"Create a quadrant diagram for project evaluation\"\n\n"
        "Expected Output:\n{\"code\": \"quadrantChart\\n title Reach and engagement of campaigns\\n x-axis Low Reach --> High Reach\\n y-axis Low Engagement --> High Engagement\\n quadrant-1 We should expand\\n quadrant-2 Need to promote\\n quadrant-3 Re-evaluate\\n quadrant-4 May be improved\\n Campaign A: [0.3, 0.6]\\n Campaign B: [0.45, 0.23]\\n Campaign C: [0.57, 0.69]\\n Campaign D: [0.78, 0.34]\\n Campaign E: [0.40, 0.34]\\n Campaign F: [0.35, 0.78]\"}\n\n"
        "16. Requirement Diagrams:\nExample Query:\n\"Create a requirement diagram for system specifications\"\n\n"
        "Expected Output:\n{\"code\":\"requirementDiagram\\n    requirement req1 {\\n      id: 1\\n      text: \\\"System shall support user authentication\\\"\\n    }\"}\n\n"
        "17. User Journey Diagrams:\nExample Query:\n\"Create a user journey diagram for onboarding new users\"\n\n"
        "Expected Output:\n{\"code\":\"journey\\n    title User Onboarding\\n    section Registration\\n      Click Sign Up: 5: User\\n      Fill Form: 3: User\\n      Confirm Email: 2: System\"}\n\n"
        "18. XY Diagrams:\nExample Query:\n\"Create an XY diagram for plotting data points\"\n\n"
        "Expected Output:\n{\"code\":\"xyChart\\n    xAxis label: \\\"Time\\\"\\n    yAxis label: \\\"Value\\\"\\n    data: [ [0, 1], [1, 2], [2, 3] ]\"}\n\n"
        "RULES:\n1. Use ONLY double quotes.\n2. Escape newlines with \\n.\n3. Never use markdown formatting.\n4. Always output STRICT JSON with proper syntax as shown in the examples."
    )

    try:
        user_input = state.get("input", "")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])

        # Get the raw response content.
        raw = response.content.strip()

        # Remove any markdown code fences if present.
        cleaned = re.sub(r'^```(json)?|```$', '', raw, flags=re.MULTILINE | re.IGNORECASE)

        # Replace single quotes around keys with double quotes (if any).
        cleaned = re.sub(r"'(?=\s*:)", '"', cleaned)
        
        # Extract the first JSON object from the cleaned string.
        json_match = re.search(r'({.*})', cleaned, re.DOTALL)
        if not json_match:
            return {"error": f"Could not find valid JSON in response:\n{cleaned}"}
        json_str = json_match.group(1)

        # Parse JSON.
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"error": f"JSON Error: {str(e)}\nExtracted JSON: {json_str}"}

        # Post-process: Replace escaped newlines with actual newlines and fix known syntax issues.
        if 'code' in data:
            data['code'] = data['code'].replace('\\n', '\n')
            data['code'] = post_process_code(data['code'])
            # Validate that we have valid starting syntax for supported diagram types.
            if not re.search(
                r'^\s*(graph|classDiagram|sequenceDiagram|erDiagram|gantt|mindmap|stateDiagram-v2|timeline|gitGraph|C4Context|sankey|flowchart|pie|quadrantChart|requirementDiagram|journey|xyChart)', 
                data['code']
            ):
                return {"error": "Invalid diagram syntax"}
            
        return data

    except Exception as e:
        return {"error": f"Processing Error: {str(e)}"}

graph = lg.Graph()
graph.add_node("process", process_query)
graph.set_entry_point("process")
graph.set_finish_point("process")
flowchart_workflow = graph.compile()