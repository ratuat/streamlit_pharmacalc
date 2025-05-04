import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from openai import OpenAI
import json

# Initialize OpenAI client

api_key = st.secrets["deepseek"]["api_key"]
client = OpenAI(api_key, 
                base_url="https://api.deepseek.com")

# Function to fetch parameters from OpenAI API
def fetch_parameters_from_api(country, drug_name, drug_comparator, disease):
    prompt =        f"""
                      You are an expert in pharmacoeconomics. Provide realistic parameters for a pharmacoeconomic analysis of a drug named '{drug_name}' used to treat '{disease}' in '{country}, drug to comapre with '{drug_comparator}'.

                      Respond ONLY with a valid JSON object and NOTHING else. DO NOT include markdown formatting, code blocks, or any prefix like 'json'. Your response MUST begin with '{{' and end with '}}'.

                      The JSON must include the following structure with realistic placeholder values (fill all fields):

                      {{
                        "CEA": {{
                          "C_B": float,
                          "C_A": float,
                          "E_B": float,
                          "E_A": float,
                          "lambda": float
                        }},
                        "CUA": {{
                          "utility_B": [float, float],
                          "time_B": [float, float],
                          "utility_A": [float, float],
                          "time_A": [float, float],
                          "C_B": float,
                          "C_A": float
                        }},
                        "CBA": {{
                          "benefits_B": float,
                          "benefits_A": float,
                          "C_B": float,
                          "C_A": float
                        }},
                        "Decision_Analysis": {{
                          "probabilities": [float, float],
                          "costs": [float, float],
                          "lambda": float,
                          "E_B": float,
                          "E_A": float,
                          "C_B": float,
                          "C_A": float
                        }},
                        "Markov_Modeling": {{
                          "proportions": [float, float],
                          "state_costs": [float, float],
                          "state_outcomes": [float, float]
                        }},
                        "Discounting": {{
                          "future_value": float,
                          "discount_rate": float,
                          "time": float
                        }},
                        "Sensitivity_Analysis": {{
                          "param_min": float,
                          "param_max": float
                        }}
                      }}

                    Values must be realistic for {country} and the treatment of {disease} with {drug_name}, drug comparator {drug_comparator}. If you make assumptions, reflect them in the values, but DO NOT include explanatory text.
                    """

    try:
        
        response = client.chat.completions.create(
                                                    model="deepseek-chat",
                                                    messages=[
                                                        {"role": "system", "content": "You are a pharmacoeconomic expert providing structured JSON data."},
                                                        {"role": "user", "content": prompt},
                                                    ],
                                                    stream=False,
                                                    temperature=0.3,
                                                    max_tokens=1500
                                                )

        # Parse JSON response
        params = json.loads(response.choices[0].message.content)
        return params
    except Exception as e:
        st.error(f"Ошибка API: {e}. Используются параметры по умолчанию.")
        # Fallback parameters
        return {
            "CEA": {"C_B": 50000, "C_A": 30000, "E_B": 5, "E_A": 3, "lambda": 50000},
            "CUA": {"utility_B": [0.8, 0.6], "time_B": [2, 3], "utility_A": [0.7, 0.5], "time_A": [2, 3], "C_B": 50000, "C_A": 30000},
            "CBA": {"benefits_B": 80000, "benefits_A": 60000, "C_B": 50000, "C_A": 30000},
            "Decision_Analysis": {"probabilities": [0.6, 0.4], "costs": [20000, 40000], "lambda": 50000, "E_B": 5, "E_A": 3, "C_B": 50000, "C_A": 30000},
            "Markov_Modeling": {"proportions": [0.7, 0.3], "state_costs": [15000, 25000], "state_outcomes": [0.8, 0.5]},
            "Discounting": {"future_value": 10000, "discount_rate": 0.03, "time": 5},
            "Sensitivity_Analysis": {"param_min": 20000, "param_max": 60000}
        }
        

# Calculation functions (unchanged)
def calculate_icer(C_B, C_A, E_B, E_A):
    delta_C = C_B - C_A
    delta_E = E_B - E_A
    return delta_C / delta_E if delta_E != 0 else float('inf')

def calculate_inb(C_B, C_A, E_B, E_A, lambda_):
    delta_C = C_B - C_A
    delta_E = E_B - E_A
    return lambda_ * delta_E - delta_C

def calculate_qaly(utility_weights, times):
    return sum(u * t for u, t in zip(utility_weights, times))

def calculate_icer_cua(C_B, C_A, QALY_B, QALY_A):
    delta_C = C_B - C_A
    delta_QALY = QALY_B - QALY_A
    return delta_C / delta_QALY if delta_QALY != 0 else float('inf')

def calculate_net_benefit(benefits, costs):
    return benefits - costs

def calculate_benefit_cost_ratio(benefits, costs):
    return benefits / costs if costs != 0 else float('inf')

def calculate_expected_cost(probabilities, costs):
    return sum(p * c for p, c in zip(probabilities, costs))

def calculate_markov_costs(proportions, state_costs):
    return sum(p * c for p, c in zip(proportions, state_costs))

def calculate_markov_outcomes(proportions, state_outcomes):
    return sum(p * o for p, o in zip(proportions, state_outcomes))

def calculate_discounted_value(future_value, discount_rate, time):
    return future_value / (1 + discount_rate) ** time

def calculate_sensitivity_range(param_min, param_max, base_C_A, E_B, E_A):
    icer_min = (param_min - base_C_A) / (E_B - E_A) if (E_B - E_A) != 0 else float('inf')
    icer_max = (param_max - base_C_A) / (E_B - E_A) if (E_B - E_A) != 0 else float('inf')
    return icer_min, icer_max

# Streamlit app
st.set_page_config(layout="wide")

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'
if 'parameters' not in st.session_state:
    st.session_state.parameters = None

# Stage 1: Input Form
if st.session_state.stage == 'input':
    st.title("Фармакоэкономический анализ")
    with st.form("input_form"):
        country = st.text_input("Страна проведения исследования")
        drug_name = st.text_input("Наименование препарата")
        disease = st.text_input("Тип болезни")
        drug_comparator = st.text_input("Препарат, с которым производится сравнение")
        
        submitted = st.form_submit_button("Далее")
        if submitted and country and drug_name and disease:
            st.session_state.parameters = fetch_parameters_from_api(country, drug_name, drug_comparator, disease)
            st.session_state.stage = 'analysis'
            st.rerun()

# Stage 2: Analysis Interface
if st.session_state.stage == 'analysis':
    st.title("Фармакоэкономический калькулятор")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Параметры")
        
        with st.expander("Анализ эффективности затрат (CEA)"):
            cea_params = st.session_state.parameters["CEA"]
            cea_C_B = st.number_input("Затраты B ($)", value=float(cea_params["C_B"]), key="cea_C_B")
            cea_C_A = st.number_input("Затраты A ($)", value=float(cea_params["C_A"]), key="cea_C_A")
            cea_E_B = st.number_input("Эффективность B (годы жизни)", value=float(cea_params["E_B"]), key="cea_E_B")
            cea_E_A = st.number_input("Эффективность A (годы жизни)", value=float(cea_params["E_A"]), key="cea_E_A")
            cea_lambda = st.number_input("Порог готовности платить ($/единица)", value=float(cea_params["lambda"]), key="cea_lambda")
        
        with st.expander("Анализ полезности затрат (CUA)"):
            cua_params = st.session_state.parameters["CUA"]
            cua_C_B = st.number_input("Затраты B ($)", value=float(cua_params["C_B"]), key="cua_C_B")
            cua_C_A = st.number_input("Затраты A ($)", value=float(cua_params["C_A"]), key="cua_C_A")
            cua_utility_B = st.text_input("Веса полезности B (через запятую)", value=",".join(map(str, cua_params["utility_B"])), key="cua_utility_B")
            cua_time_B = st.text_input("Время B (годы, через запятую)", value=",".join(map(str, cua_params["time_B"])), key="cua_time_B")
            cua_utility_A = st.text_input("Веса полезности A (через запятую)", value=",".join(map(str, cua_params["utility_A"])), key="cua_utility_A")
            cua_time_A = st.text_input("Время A (годы, через запятую)", value=",".join(map(str, cua_params["time_A"])), key="cua_time_A")
        
        with st.expander("Анализ затрат и выгод (CBA)"):
            cba_params = st.session_state.parameters["CBA"]
            cba_benefits_B = st.number_input("Выгоды B ($)", value=float(cba_params["benefits_B"]), key="cba_benefits_B")
            cba_benefits_A = st.number_input("Выгоды A ($)", value=float(cba_params["benefits_A"]), key="cba_benefits_A")
            cba_C_B = st.number_input("Затраты B ($)", value=float(cba_params["C_B"]), key="cba_C_B")
            cba_C_A = st.number_input("Затраты A ($)", value=float(cba_params["C_A"]), key="cba_C_A")
        
        with st.expander("Анализ решений"):
            da_params = st.session_state.parameters["Decision_Analysis"]
            da_probabilities = st.text_input("Вероятности (через запятую)", value=",".join(map(str, da_params["probabilities"])), key="da_probabilities")
            da_costs = st.text_input("Затраты на исход ($, через запятую)", value=",".join(map(str, da_params["costs"])), key="da_costs")
            da_lambda = st.number_input("Порог готовности платить ($/единица)", value=float(da_params["lambda"]), key="da_lambda")
            da_E_B = st.number_input("Эффективность B", value=float(da_params["E_B"]), key="da_E_B")
            da_E_A = st.number_input("Эффективность A", value=float(da_params["E_A"]), key="da_E_A")
            da_C_B = st.number_input("Затраты B ($)", value=float(da_params["C_B"]), key="da_C_B")
            da_C_A = st.number_input("Затраты A ($)", value=float(da_params["C_A"]), key="da_C_A")
        
        with st.expander("Моделирование Маркова"):
            markov_params = st.session_state.parameters["Markov_Modeling"]
            markov_proportions = st.text_input("Доли в состояниях (через запятую)", value=",".join(map(str, markov_params["proportions"])), key="markov_proportions")
            markov_state_costs = st.text_input("Затраты на состояния ($, через запятую)", value=",".join(map(str, markov_params["state_costs"])), key="markov_state_costs")
            markov_state_outcomes = st.text_input("Результаты состояний (через запятую)", value=",".join(map(str, markov_params["state_outcomes"])), key="markov_state_outcomes")
        
        with st.expander("Дисконтирование"):
            disc_params = st.session_state.parameters["Discounting"]
            disc_future_value = st.number_input("Будущая стоимость ($)", value=float(disc_params["future_value"]), key="disc_future_value")
            disc_rate = st.number_input("Ставка дисконтирования", value=float(disc_params["discount_rate"]), key="disc_rate")
            disc_time = st.number_input("Время (годы)", value=float(disc_params["time"]), key="disc_time")
        
        with st.expander("Анализ чувствительности"):
            sens_params = st.session_state.parameters["Sensitivity_Analysis"]
            sens_param_min = st.number_input("Минимальное значение параметра ($)", value=float(sens_params["param_min"]), key="sens_param_min")
            sens_param_max = st.number_input("Максимальное значение параметра ($)", value=float(sens_params["param_max"]), key="sens_param_max")

    # Main area: Results and Dashboards
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Результаты расчетов")
        
        st.subheader("Анализ эффективности затрат (CEA)")
        icer = calculate_icer(cea_C_B, cea_C_A, cea_E_B, cea_E_A)
        inb = calculate_inb(cea_C_B, cea_C_A, cea_E_B, cea_E_A, cea_lambda)
        st.write(f"**ICER**: ${icer:,.2f} за единицу эффективности")
        st.write(f"**INB**: ${inb:,.2f}")
        st.write("**Пояснение**: ICER показывает дополнительные затраты на единицу эффективности. INB отражает чистую денежную выгоду при заданном пороге готовности платить.")
        
        st.subheader("Анализ полезности затрат (CUA)")
        try:
            utility_B = [float(x) for x in cua_utility_B.split(",")]
            time_B = [float(x) for x in cua_time_B.split(",")]
            utility_A = [float(x) for x in cua_utility_A.split(",")]
            time_A = [float(x) for x in cua_time_A.split(",")]
            qaly_B = calculate_qaly(utility_B, time_B)
            qaly_A = calculate_qaly(utility_A, time_A)
            icer_cua = calculate_icer_cua(cua_C_B, cua_C_A, qaly_B, qaly_A)
            st.write(f"**QALY B**: {qaly_B:.2f}")
            st.write(f"**QALY A**: {qaly_A:.2f}")
            st.write(f"**ICER (CUA)**: ${icer_cua:,.2f} за QALY")
            st.write("**Пояснение**: QALY объединяет качество и продолжительность жизни. ICER для CUA оценивает затраты на дополнительный QALY.")
        except:
            st.error("Ошибка в формате ввода для CUA. Убедитесь, что списки разделены запятыми и содержат числа.")
        
        st.subheader("Анализ затрат и выгод (CBA)")
        net_benefit_B = calculate_net_benefit(cba_benefits_B, cba_C_B)
        net_benefit_A = calculate_net_benefit(cba_benefits_A, cba_C_A)
        bc_ratio_B = calculate_benefit_cost_ratio(cba_benefits_B, cba_C_B)
        bc_ratio_A = calculate_benefit_cost_ratio(cba_benefits_A, cba_C_A)
        st.write(f"**Чистая выгода B**: ${net_benefit_B:,.2f}")
        st.write(f"**Чистая выгода A**: ${net_benefit_A:,.2f}")
        st.write(f"**Соотношение выгоды к затратам B**: {bc_ratio_B:.2f}")
        st.write(f"**Соотношение выгоды к затратам A**: {bc_ratio_A:.2f}")
        st.write("**Пояснение**: Чистая выгода показывает превышение выгод над затратами. Соотношение >1 указывает на экономическую выгоду.")
        
        st.subheader("Анализ решений")
        try:
            probabilities = [float(x) for x in da_probabilities.split(",")]
            costs = [float(x) for x in da_costs.split(",")]
            expected_cost = calculate_expected_cost(probabilities, costs)
            da_inb = calculate_inb(da_C_B, da_C_A, da_E_B, da_E_A, da_lambda)
            st.write(f"**Ожидаемые затраты**: ${expected_cost:,.2f}")
            st.write(f"**INB**: ${da_inb:,.2f}")
            st.write("**Пояснение**: Ожидаемые затраты учитывают вероятности исходов. INB сравнивает варианты лечения.")
        except:
            st.error("Ошибка в формате ввода для анализа решений.")
        
        st.subheader("Моделирование Маркова")
        try:
            proportions = [float(x) for x in markov_proportions.split(",")]
            state_costs = [float(x) for x in markov_state_costs.split(",")]
            state_outcomes = [float(x) for x in markov_state_outcomes.split(",")]
            markov_costs = calculate_markov_costs(proportions, state_costs)
            markov_outcomes = calculate_markov_outcomes(proportions, state_outcomes)
            st.write(f"**Ожидаемые затраты**: ${markov_costs:,.2f}")
            st.write(f"**Ожидаемые результаты**: {markov_outcomes:.2f}")
            st.write("**Пояснение**: Рассчитывает долгосрочные затраты и результаты с учетом переходов между состояниями.")
        except:
            st.error("Ошибка в формате ввода для моделирования Маркова.")
        
        st.subheader("Дисконтирование")
        discounted_value = calculate_discounted_value(disc_future_value, disc_rate, disc_time)
        st.write(f"**Приведенная стоимость**: ${discounted_value:,.2f}")
        st.write("**Пояснение**: Корректирует будущие затраты или результаты на временную стоимость денег.")
        
        st.subheader("Анализ чувствительности")
        icer_min, icer_max = calculate_sensitivity_range(sens_param_min, sens_param_max, cea_C_A, cea_E_B, cea_E_A)
        st.write(f"**Диапазон ICER**: от ${icer_min:,.2f} до ${icer_max:,.2f}")
        st.write("**Пояснение**: Показывает, как изменения в параметрах влияют на ICER.")

    with col2:
        st.header("Дашборды")
        
        cea_data = pd.DataFrame({
            "Метрика": ["ICER", "INB"],
            "Значение": [icer, inb]
        })
        fig_cea = px.bar(cea_data, x="Метрика", y="Значение", title="CEA: ICER и INB")
        st.plotly_chart(fig_cea, use_container_width=True)
        
        try:
            cua_data = pd.DataFrame({
                "Вмешательство": ["B", "A"],
                "QALY": [qaly_B, qaly_A]
            })
            fig_cua = px.bar(cua_data, x="Вмешательство", y="QALY", title="CUA: QALY")
            st.plotly_chart(fig_cua, use_container_width=True)
        except:
            st.write("Дашборд для CUA недоступен из-за ошибки ввода.")
        
        sens_data = pd.DataFrame({
            "Параметр": ["Затраты B"],
            "Мин": [icer_min],
            "Макс": [icer_max]
        })
        fig_sens = px.bar(sens_data, y="Параметр", x=["Мин", "Макс"], title="Анализ чувствительности: Диапазон ICER", orientation='h')
        st.plotly_chart(fig_sens, use_container_width=True)
