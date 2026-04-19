"""
agent.py — Physics Study Buddy Agent
Agentic AI Capstone 2026 | Dr. Kanthi Kiran Sirra

This module builds and returns the compiled LangGraph agent, the embedder,
and the ChromaDB collection. Import build_agent() from capstone_streamlit.py
and from the notebook.
"""

import os
import math
from typing import TypedDict, List

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    user_name:    str


# ─────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": """
Newton's Laws of Motion are three fundamental principles that describe the relationship between
a body and the forces acting upon it.

First Law (Law of Inertia): An object at rest stays at rest and an object in motion stays in
motion with the same speed and direction unless acted upon by an unbalanced external force.
This property of matter is called inertia. Example: A ball rolling on a smooth floor continues
to roll until friction or a wall stops it.

Second Law (Law of Acceleration): The acceleration of an object is directly proportional to the
net force acting on it and inversely proportional to its mass. The formula is F = ma, where F is
the net force in Newtons (N), m is mass in kilograms (kg), and a is acceleration in m/s².
Example: A 5 kg object with a net force of 20 N will accelerate at 4 m/s².

Third Law (Law of Action-Reaction): For every action, there is an equal and opposite reaction.
When object A exerts a force on object B, object B exerts an equal and opposite force on object A.
Example: When you push a wall, the wall pushes back on you with the same force.

Key formulas:
- F = ma (Force = mass × acceleration)
- Weight W = mg (where g = 9.8 m/s² on Earth)
- Momentum p = mv
- Impulse J = F × t = change in momentum

Applications: vehicle design, rocket propulsion, sports biomechanics, and everyday motion analysis.
"""
    },
    {
        "id": "doc_002",
        "topic": "Work, Energy and Power",
        "text": """
Work, Energy, and Power are closely related concepts in physics that describe how forces
interact with objects over time and distance.

Work (W): Work is done when a force causes displacement in the direction of the force.
Formula: W = F × d × cos(θ), where F is force in Newtons, d is displacement in metres,
and θ is the angle between force and displacement. Unit: Joule (J). If θ = 0°, W = F × d.
No work is done if displacement is zero or perpendicular to force.

Kinetic Energy (KE): Energy possessed by a moving object. Formula: KE = ½mv², where m is
mass in kg and v is speed in m/s. Unit: Joule (J). A 2 kg object moving at 3 m/s has KE =
½ × 2 × 9 = 9 J.

Potential Energy (PE): Energy stored in an object due to its position or configuration.
Gravitational PE = mgh, where m = mass, g = 9.8 m/s², h = height above reference.
Elastic PE = ½kx², where k = spring constant and x = compression/extension.

Work-Energy Theorem: The net work done on an object equals the change in its kinetic energy.
W_net = ΔKE = ½mv² - ½mu²

Conservation of Energy: Energy cannot be created or destroyed, only transformed.
Total mechanical energy = KE + PE = constant (in absence of non-conservative forces).

Power (P): Rate of doing work. Formula: P = W/t = F × v. Unit: Watt (W) = J/s.
1 horsepower = 746 Watts.

Efficiency: η = (Useful output energy / Total input energy) × 100%
"""
    },
    {
        "id": "doc_003",
        "topic": "Laws of Thermodynamics",
        "text": """
Thermodynamics is the branch of physics dealing with heat, work, and energy transformations
in systems. There are four fundamental laws.

Zeroth Law: If two systems are each in thermal equilibrium with a third system, they are in
thermal equilibrium with each other. This defines temperature as a measurable property.

First Law (Conservation of Energy): Energy cannot be created or destroyed. For a thermodynamic
system: ΔU = Q - W, where ΔU is change in internal energy, Q is heat added to the system,
and W is work done by the system. If heat is added and no work is done, internal energy increases.

Second Law: Heat flows naturally from a hot object to a cold object. No heat engine can be
100% efficient. Entropy (disorder) of an isolated system always increases or stays the same.
Entropy S is measured in J/K. ΔS = Q/T for reversible processes.

Third Law: The entropy of a perfect crystal at absolute zero (0 K) is zero. This means you
cannot reach absolute zero temperature in a finite number of steps.

Key processes:
- Isothermal: Temperature constant (ΔU = 0, Q = W)
- Adiabatic: No heat exchange (Q = 0, ΔU = -W)
- Isobaric: Pressure constant (W = PΔV)
- Isochoric: Volume constant (W = 0, ΔU = Q)

Carnot Efficiency: η = 1 - (T_cold / T_hot), where temperatures are in Kelvin.
This is the maximum possible efficiency of any heat engine.
"""
    },
    {
        "id": "doc_004",
        "topic": "Electric Current and Ohm's Law",
        "text": """
Electric current is the flow of electric charge through a conductor. It is fundamental to
all electrical circuits and devices.

Electric Current (I): Rate of flow of charge. Formula: I = Q/t, where Q is charge in Coulombs
and t is time in seconds. Unit: Ampere (A). Conventional current flows from positive to negative
terminal; electrons flow in the opposite direction.

Voltage (V): Electric potential difference between two points. It is the energy per unit charge.
Unit: Volt (V). EMF (electromotive force) is the voltage provided by a battery or source.

Resistance (R): Opposition to the flow of current. Unit: Ohm (Ω).
Resistance depends on material (resistivity ρ), length L, and cross-sectional area A:
R = ρL/A. Longer wires have higher resistance; thicker wires have lower resistance.

Ohm's Law: At constant temperature, the current through a conductor is directly proportional
to the voltage across it. V = IR. This gives: I = V/R and R = V/I.
Example: A 12V battery connected to a 4Ω resistor gives I = 12/4 = 3A.

Power in circuits: P = VI = I²R = V²/R. Unit: Watt.

Resistors in Series: R_total = R1 + R2 + R3 (current same, voltage divides)
Resistors in Parallel: 1/R_total = 1/R1 + 1/R2 + 1/R3 (voltage same, current divides)

Kirchhoff's Laws:
- KCL (Current): Sum of currents entering a node = sum leaving it.
- KVL (Voltage): Sum of all voltages around any closed loop = 0.
"""
    },
    {
        "id": "doc_005",
        "topic": "Capacitors and Capacitance",
        "text": """
A capacitor is a device that stores electrical energy in an electric field between two
conducting plates separated by an insulator (dielectric).

Capacitance (C): Ability of a capacitor to store charge. Formula: C = Q/V, where Q is charge
stored in Coulombs and V is the voltage across the capacitor. Unit: Farad (F).
Practical capacitors are in microfarads (μF) or picofarads (pF).

Parallel Plate Capacitor: C = ε₀ × εr × A/d, where ε₀ = 8.85 × 10⁻¹² F/m (permittivity of
free space), εr = relative permittivity of dielectric, A = area of plates, d = separation distance.
Larger plate area or smaller separation increases capacitance.

Energy stored: E = ½CV² = ½QV = Q²/2C. A capacitor charged to 10V with C = 2F stores
E = ½ × 2 × 100 = 100 J.

Capacitors in Series: 1/C_total = 1/C1 + 1/C2 + 1/C3 (voltage divides, charge same)
Capacitors in Parallel: C_total = C1 + C2 + C3 (voltage same, charge divides)

Charging and Discharging: When connected to a battery through a resistor, a capacitor charges
exponentially. Time constant τ = RC. After time τ, capacitor reaches ~63% of full charge.
After 5τ, it is considered fully charged.

Dielectric: An insulating material placed between the plates that increases capacitance by
reducing the electric field. Common dielectrics: air (εr = 1), paper (εr ≈ 3.5), ceramic (εr ≈ 6-10000).

Applications: filtering in power supplies, timing circuits, camera flash, memory storage.
"""
    },
    {
        "id": "doc_006",
        "topic": "Magnetic Force and Faraday's Law",
        "text": """
Magnetism and electromagnetism describe the relationship between electric currents and
magnetic fields.

Magnetic Field (B): A region around a magnet or current-carrying conductor where magnetic
force acts. Unit: Tesla (T). Earth's magnetic field is about 25-65 μT.

Force on a Moving Charge: F = qvB sin(θ), where q = charge, v = velocity, B = magnetic field,
θ = angle between v and B. Maximum force when θ = 90°. No force when charge moves parallel
to field. Direction given by the right-hand rule or Fleming's left-hand rule for motors.

Force on a Current-Carrying Conductor: F = BIL sin(θ), where I = current, L = length of
conductor in the field. This principle is used in electric motors.

Biot-Savart Law: Magnetic field due to a current element. For a long straight wire:
B = μ₀I / (2πr), where μ₀ = 4π × 10⁻⁷ T·m/A and r = distance from wire.

Faraday's Law of Electromagnetic Induction: A changing magnetic flux through a loop induces
an EMF. EMF = -dΦ/dt, where Φ = B × A × cos(θ) is the magnetic flux in Weber (Wb).
The negative sign indicates Lenz's Law — the induced current opposes the change causing it.

Lenz's Law: The direction of induced current is such that it opposes the change in flux
that caused it. This is a consequence of energy conservation.

Applications: Electric generators convert mechanical energy to electrical energy using Faraday's
Law. Transformers use mutual induction. MRI machines use strong magnetic fields.
"""
    },
    {
        "id": "doc_007",
        "topic": "Wave Motion and Sound",
        "text": """
Waves are disturbances that transfer energy through a medium (or vacuum) without transferring matter.

Types of Waves:
- Transverse waves: Displacement perpendicular to direction of propagation (light, water waves).
  They have crests and troughs.
- Longitudinal waves: Displacement parallel to direction of propagation (sound waves).
  They have compressions and rarefactions.

Key wave properties:
- Amplitude (A): Maximum displacement from equilibrium. Related to energy.
- Wavelength (λ): Distance between two consecutive similar points (e.g., crest to crest). Unit: metre.
- Frequency (f): Number of complete oscillations per second. Unit: Hertz (Hz).
- Time Period (T): Time for one complete oscillation. T = 1/f.
- Wave speed (v): v = fλ = λ/T.

Sound Waves: Longitudinal mechanical waves that require a medium. Speed of sound in air at
20°C ≈ 343 m/s. Speed is higher in solids than liquids than gases.
Pitch is determined by frequency; loudness by amplitude.

Doppler Effect: Change in observed frequency when source or observer is moving.
If source moves toward observer: observed frequency increases.
Formula: f' = f × (v ± v_observer) / (v ∓ v_source).

Superposition and Interference:
- Constructive interference: waves in phase, amplitudes add.
- Destructive interference: waves out of phase, amplitudes cancel.

Standing Waves: Formed by superposition of two identical waves travelling in opposite directions.
For a string fixed at both ends: λ_n = 2L/n, f_n = nv/2L.
"""
    },
    {
        "id": "doc_008",
        "topic": "Optics — Reflection and Refraction",
        "text": """
Optics is the study of the behaviour and properties of light and its interactions with matter.

Reflection: When light bounces off a surface.
Laws of Reflection:
1. Angle of incidence = Angle of reflection (both measured from the normal).
2. Incident ray, reflected ray, and normal all lie in the same plane.
Mirrors: Plane mirrors produce virtual, erect, same-sized images. Concave mirrors can produce
real or virtual images depending on object position. Convex mirrors always produce virtual, erect,
diminished images — used in rear-view mirrors.
Mirror Formula: 1/v + 1/u = 1/f, where v = image distance, u = object distance, f = focal length.
Magnification m = -v/u.

Refraction: Bending of light when it passes from one medium to another due to change in speed.
Snell's Law: n₁ sin(θ₁) = n₂ sin(θ₂), where n is refractive index.
Refractive index n = speed of light in vacuum / speed of light in medium = c/v.
n for glass ≈ 1.5, water ≈ 1.33, air ≈ 1.0003.

Total Internal Reflection: Occurs when light travels from denser to rarer medium and angle of
incidence exceeds critical angle. Formula: sin(θ_c) = n2/n1. Used in optical fibres and diamonds.

Lenses:
- Convex (converging) lens: Focuses light. Used in cameras, projectors, the eye.
- Concave (diverging) lens: Spreads light. Used to correct myopia.
Lens Formula: 1/v - 1/u = 1/f. Power of lens P = 1/f (in metres). Unit: Dioptre (D).

Dispersion: Splitting of white light into spectrum by a prism because different colours
have different refractive indices. Violet bends most, red bends least.
"""
    },
    {
        "id": "doc_009",
        "topic": "Modern Physics — Photoelectric Effect",
        "text": """
Modern physics introduced quantum theory and explained phenomena that classical physics could not.

Photoelectric Effect: When light of sufficient frequency falls on a metal surface, electrons
are emitted. This was explained by Einstein in 1905, for which he received the Nobel Prize.

Key observations:
1. Electrons are emitted only if frequency of light exceeds a threshold frequency (f₀) —
   specific to each metal. Below f₀, no electrons are emitted regardless of intensity.
2. Maximum kinetic energy of emitted electrons depends on frequency, not intensity of light.
3. Number of electrons emitted depends on intensity of light.
4. Emission is instantaneous — no time delay.

Einstein's Equation: KE_max = hf - φ, where h = Planck's constant = 6.626 × 10⁻³⁴ J·s,
f = frequency of incident light, φ = work function (energy needed to free an electron from metal).
Threshold frequency: f₀ = φ/h. Stopping potential: eV₀ = KE_max.

Wave-Particle Duality: Light behaves as both a wave (interference, diffraction) and a particle
(photoelectric effect, Compton scattering). de Broglie proposed matter also has wave properties:
λ = h/mv (de Broglie wavelength).

Atomic Models:
- Rutherford: Nucleus at centre, electrons orbit like planets.
- Bohr Model: Electrons orbit in fixed energy levels. Energy of nth level in hydrogen:
  E_n = -13.6/n² eV. When electron jumps between levels, photon is emitted or absorbed.

Radioactivity: Unstable nuclei emit alpha (α), beta (β), or gamma (γ) radiation.
Half-life T½: Time for half the radioactive atoms to decay. N = N₀ × (1/2)^(t/T½).
"""
    },
    {
        "id": "doc_010",
        "topic": "Gravitation and Kepler's Laws",
        "text": """
Gravitation describes the attractive force between any two objects with mass. It governs the
motion of planets, satellites, and celestial bodies.

Newton's Law of Universal Gravitation: Every object attracts every other object with a force
proportional to the product of their masses and inversely proportional to the square of the
distance between them. F = G × m₁ × m₂ / r², where G = 6.674 × 10⁻¹¹ N·m²/kg²
(universal gravitational constant), m₁ and m₂ are masses in kg, r is distance between centres in metres.

Gravitational Field Strength (g): g = GM/r². On Earth's surface g ≈ 9.8 m/s².
Weight W = mg. Mass is constant; weight changes with location.

Gravitational Potential Energy: PE = -GMm/r (negative because gravitational force is attractive).
Near Earth's surface: PE = mgh.

Escape Velocity: Minimum speed to escape a planet's gravitational field.
v_escape = √(2GM/R) = √(2gR). For Earth: v_escape ≈ 11.2 km/s.

Orbital Velocity: Speed needed for circular orbit. v_orbit = √(GM/r).
For low Earth orbit: v_orbit ≈ 7.9 km/s.

Kepler's Laws of Planetary Motion:
1. Law of Orbits: All planets move in elliptical orbits with the Sun at one focus.
2. Law of Areas: A line from planet to Sun sweeps equal areas in equal times
   (planets move faster when closer to Sun).
3. Law of Periods: T² ∝ r³, where T is orbital period and r is average orbital radius.
   Precisely: T² = (4π²/GM) × r³.

Satellites: Geostationary satellites orbit at ~36,000 km with T = 24 hours, appearing
stationary relative to Earth — used for communication and weather observation.
"""
    }
]


# ─────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────

def _make_nodes(llm, embedder, collection):
    """Return all 8 node functions closed over llm, embedder, collection."""

    def memory_node(state: CapstoneState) -> dict:
        messages  = state.get("messages", [])
        question  = state["question"]
        user_name = state.get("user_name", "")
        lower_q   = question.lower()
        if "my name is" in lower_q:
            parts = lower_q.split("my name is")
            if len(parts) > 1:
                candidate = parts[1].strip().split()
                if candidate:
                    user_name = candidate[0].capitalize()
        messages = messages + [{"role": "user", "content": question}]
        messages = messages[-6:]
        return {"messages": messages, "user_name": user_name}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        prompt = f"""You are a routing assistant for a Physics Study Buddy chatbot.
Classify the student's question into EXACTLY ONE of these routes:

- retrieve: The question asks about a physics concept, formula, law, or topic
  that can be answered from the knowledge base (Newton's laws, thermodynamics,
  optics, waves, capacitors, magnetism, photoelectric effect, gravitation, etc.).

- tool: The question requires a calculation or arithmetic
  (e.g., "calculate the force", "compute kinetic energy", "what is 5 * 3").

- memory_only: The question is a greeting, small talk, or asks about something
  already discussed (e.g., "hello", "thanks", "what did I just ask").

Student question: {question}

Reply with ONE word only: retrieve, tool, or memory_only"""
        response = llm.invoke(prompt)
        route = response.content.strip().lower().split()[0]
        if route not in ["retrieve", "tool", "memory_only"]:
            route = "retrieve"
        return {"route": route}

    def retrieval_node(state: CapstoneState) -> dict:
        question        = state["question"]
        query_embedding = embedder.encode([question]).tolist()
        results         = collection.query(query_embeddings=query_embedding, n_results=3)
        context_parts, sources = [], []
        for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
            topic = meta["topic"]
            context_parts.append(f"[{topic}]\n{doc_text.strip()}")
            sources.append(topic)
        retrieved = "\n\n".join(context_parts)
        return {"retrieved": retrieved, "sources": sources}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def safe_calculate(expression: str) -> str:
        try:
            safe_dict = {
                "__builtins__": {},
                "sqrt": math.sqrt, "pow": math.pow, "abs": abs,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "pi": math.pi, "e": math.e, "log": math.log,
                "exp": math.exp, "ceil": math.ceil, "floor": math.floor
            }
            result = eval(expression, safe_dict)
            return f"Result: {result}"
        except Exception as ex:
            return f"Calculation error: {str(ex)}"

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        extract_prompt = f"""Extract ONLY the mathematical expression to calculate from this question.
Return ONLY the Python-evaluable expression (e.g., '5 * 3', '0.5 * 2 * 9**2').
If no clear calculation exists, return: none

Question: {question}"""
        try:
            response   = llm.invoke(extract_prompt)
            expression = response.content.strip()
            if expression.lower() == "none" or not expression:
                tool_result = "No numerical calculation found in the question."
            else:
                tool_result = safe_calculate(expression)
        except Exception as ex:
            tool_result = f"Tool error: {str(ex)}"
        return {"tool_result": tool_result, "retrieved": "", "sources": []}

    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        user_name    = state.get("user_name", "")
        eval_retries = state.get("eval_retries", 0)

        history = ""
        for msg in messages[-4:]:
            history += f"{msg['role'].capitalize()}: {msg['content']}\n"

        greeting         = f"Student's name is {user_name}. " if user_name else ""
        retry_instruction = ""
        if eval_retries > 0:
            retry_instruction = "\nIMPORTANT: Previous answer failed faithfulness check. Be MORE strictly grounded in the context. Do NOT add information not in the context."

        context_section = ""
        if retrieved:
            context_section += f"KNOWLEDGE BASE CONTEXT:\n{retrieved}\n"
        if tool_result:
            context_section += f"CALCULATOR RESULT:\n{tool_result}\n"

        system_prompt = f"""You are a Physics Study Buddy for B.Tech students. {greeting}
You ONLY answer from the provided context below. Do NOT use general knowledge or make up formulas.
If the context does not contain the answer, say: "I don't have information on that topic in my syllabus.
Please ask your professor or refer to your textbook."{retry_instruction}

{context_section}
CONVERSATION HISTORY:
{history}"""

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ])
        return {"answer": response.content.strip()}

    def eval_node(state: CapstoneState) -> dict:
        answer       = state.get("answer", "")
        retrieved    = state.get("retrieved", "")
        eval_retries = state.get("eval_retries", 0)

        if not retrieved:
            return {"faithfulness": 1.0, "eval_retries": eval_retries}

        eval_prompt = f"""Rate how faithfully this answer is grounded in the provided context.
Score 0.0 to 1.0:
- 1.0 = every fact comes directly from the context
- 0.7 = mostly grounded, minor additions acceptable
- 0.4 = significant information added not in the context
- 0.0 = answer completely ignores the context

CONTEXT: {retrieved[:800]}
ANSWER: {answer}

Reply with a single decimal number only (e.g., 0.85)."""

        try:
            response     = llm.invoke(eval_prompt)
            faithfulness = float(response.content.strip().split()[0])
            faithfulness = max(0.0, min(1.0, faithfulness))
        except Exception:
            faithfulness = 0.5

        return {"faithfulness": faithfulness, "eval_retries": eval_retries + 1}

    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        answer   = state.get("answer", "")
        messages = messages + [{"role": "assistant", "content": answer}]
        return {"messages": messages}

    return (memory_node, router_node, retrieval_node, skip_retrieval_node,
            tool_node, answer_node, eval_node, save_node)


# ─────────────────────────────────────────────
# Routing functions (standalone — LangGraph API requirement)
# ─────────────────────────────────────────────

def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    elif route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    faithfulness  = state.get("faithfulness", 1.0)
    eval_retries  = state.get("eval_retries", 0)
    if faithfulness >= FAITHFULNESS_THRESHOLD or eval_retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


# ─────────────────────────────────────────────
# build_agent() — main entry point
# ─────────────────────────────────────────────

def build_agent():
    """
    Build and return (compiled_app, embedder, collection).
    Called once from @st.cache_resource in Streamlit and from notebook.
    """
    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # ChromaDB
    chroma_client = chromadb.Client()
    collection    = chroma_client.create_collection(name="physics_kb")
    texts         = [d["text"]              for d in DOCUMENTS]
    ids           = [d["id"]               for d in DOCUMENTS]
    metadatas     = [{"topic": d["topic"]} for d in DOCUMENTS]
    embeddings    = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

    # Nodes
    (memory_node, router_node, retrieval_node, skip_retrieval_node,
     tool_node, answer_node, eval_node, save_node) = _make_nodes(llm, embedder, collection)

    # Graph
    graph = StateGraph(CapstoneState)
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    graph.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "skip":     "skip",
        "tool":     "tool"
    })
    graph.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",
        "save":   "save"
    })

    app = graph.compile(checkpointer=MemorySaver())
    print("✅ Physics Study Buddy agent compiled successfully.")
    return app, embedder, collection


if __name__ == "__main__":
    # Quick smoke test
    app, embedder, collection = build_agent()
    config = {"configurable": {"thread_id": "smoke_test"}}
    result = app.invoke({
        "question":     "What is Newton's second law?",
        "messages":     [],
        "route":        "",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name":    ""
    }, config=config)
    print("Answer:", result["answer"][:200])
