from flask import Flask, render_template, request, jsonify
from qa import answer_question, generate_title
from main import summarize_text, generate_title_tfidf

app = Flask(__name__)

# ---- ROUTES ----

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_txt, len_orig_text, len_summary = summarize_text(rawtext)
        title = generate_title_tfidf(rawtext)
        return render_template('summary.html', title=title, summary=summary, original_txt=original_txt, len_orig_text=len_orig_text, len_summary=len_summary)
    return render_template('index.html')

@app.route('/qa', methods=['GET', 'POST'])
def qa():
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Handle AJAX request
        data = request.get_json()
        context = data.get('context')
        question = data.get('question')

        if not context or not question:
            return jsonify({"answer": "Please provide both context and question."}), 400

        title = generate_title(context)
        answer = answer_question(context, question)

        return jsonify({"title": title, "answer": answer})

    # Handle regular form submission
    context = request.form.get('context')
    question = request.form.get('question')

    if not context or not question:
        return render_template('qa_page.html', answer="Please provide both context and question.", context=context, question=question)
    
    title = generate_title(context)
    answer = answer_question(context, question)
    
    return render_template('qa_page.html', title=title, answer=answer, context=context, question=question)


if __name__ == '__main__':
    app.run(debug=True, port=5003)
