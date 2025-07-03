from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

aq10_questions = [
    "Eu costumo notar pequenos sons quando outros não notam.",
    "Eu costumo me concentrar mais no todo do que nos pequenos detalhes.",
    "Eu acho fácil fazer mais de uma coisa ao mesmo tempo.",
    "Se houver uma interrupção, consigo voltar rapidamente ao que estava fazendo.",
    "Eu acho fácil entender o que está implícito quando alguém está falando comigo.",
    "Eu sei como perceber se alguém que está me ouvindo está ficando entediado.",
    "Quando leio uma história, acho difícil entender as intenções dos personagens.",
    "Eu gosto de colecionar informações sobre categorias de coisas (ex: tipos de carros, pássaros, trens, plantas, etc.).",
    "Eu acho fácil entender o que alguém está pensando ou sentindo só de olhar para seu rosto.",
    "Eu acho difícil entender as intenções das pessoas.",
]

score_agree = [0, 6, 7, 9]  # Index das perguntas que pontuam com Agree
score_disagree = [1, 2, 3, 4, 5, 8]  # Index das perguntas que pontuam com Disagree

response_map = {
    "definitely_agree": "Concordo totalmente",
    "slightly_agree": "Concordo um pouco",
    "slightly_disagree": "Discordo um pouco",
    "definitely_disagree": "Discordo totalmente"
}


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    respostas = {}
    score_final = None
    recomendacao = None
    if request.method == "POST":
        try:
            respostas = {f"A{i + 1}_Score": request.form.get(f"A{i + 1}_Score") for i in range(10)}
            idade = request.form.get("age")
            sexo = request.form.get("gender")
            if None in respostas.values() or idade is None or sexo is None:
                resultado = {"erro": "Por favor, preencha todas as respostas."}
            else:
                resposta_codificada = []
                for idx in range(10):
                    resp = respostas[f"A{idx + 1}_Score"]
                    if idx in score_agree:
                        resposta_codificada.append(1 if resp in ["definitely_agree", "slightly_agree"] else 0)
                    else:
                        resposta_codificada.append(1 if resp in ["slightly_disagree", "definitely_disagree"] else 0)
                score_final = sum(resposta_codificada)
                if score_final >= 6:
                    recomendacao = "Recomenda encaminhamento para avaliação especializada."
                else:
                    recomendacao = "Não há indicação clara para encaminhamento especializado."
                idade = int(idade)
                sexo = int(sexo)
                entrada_modelo = resposta_codificada + [idade, sexo]
                X = np.array(entrada_modelo).reshape(1, -1)
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
                resultado = {
                    "resultado": int(pred),
                    "probabilidade_autismo": float(prob) if prob is not None else None
                }
        except Exception as e:
            resultado = {"erro": str(e)}
    return render_template(
        "index.html",
        perguntas=aq10_questions,
        resultado=resultado,
        respostas=respostas,
        response_map=response_map,
        score_final=score_final,
        recomendacao=recomendacao
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    respostas = data.get("respostas")
    idade = data.get("idade")
    sexo = data.get("sexo")
    if not respostas or len(respostas) != 10 or idade is None or sexo is None:
        return jsonify({"erro": "Dados inválidos. Envie 10 respostas, idade e sexo."}), 400
    resposta_codificada = []
    for idx in range(10):
        resp = respostas[idx]
        if idx in score_agree:
            resposta_codificada.append(1 if resp in ["definitely_agree", "slightly_agree"] else 0)
        else:
            resposta_codificada.append(1 if resp in ["slightly_disagree", "definitely_disagree"] else 0)
    score_final = sum(resposta_codificada)
    recomendacao = (
        "Recomenda encaminhamento para avaliação especializada."
        if score_final >= 6 else
        "Não há indicação clara para encaminhamento especializado."
    )
    entrada_modelo = resposta_codificada + [int(idade), int(sexo)]
    X = np.array(entrada_modelo).reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
    return jsonify({
        "resultado": int(pred),
        "probabilidade_autismo": float(prob) if prob is not None else None,
        "pontuacao_aq10": score_final,
        "recomendacao": recomendacao
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
