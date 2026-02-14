import re
from Server.flowchart_generator import flowchart_workflow
from flask import Flask, request, jsonify, render_template
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

def validate_mermaid(code):
    return re.search(
        r'^\s*(graph|classDiagram|sequenceDiagram|erDiagram|gantt|mindmap|stateDiagram-v2|timeline|gitGraph|C4Context|sankey|flowchart|pie|quadrantChart|requirementDiagram|journey|xyChart)', 
        code, 
        re.MULTILINE
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        user_input = data.get('query', '').strip()
        if not user_input:
            return jsonify({"error": "Empty query received"}), 400

        result = flowchart_workflow.invoke({"input": user_input})
        
        if not isinstance(result, dict):
            return jsonify({"error": "Invalid response format"}), 500
            
        if 'error' in result:
            return jsonify(result), 400
            
        if not validate_mermaid(result.get('code', '')):
            return jsonify({"error": "Invalid Mermaid syntax"}), 400
                
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)