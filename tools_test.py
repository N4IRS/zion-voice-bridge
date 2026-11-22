import json, urllib.request, itertools

TOOLS = [
 {"type":"function","function":{"name":"get_clock","description":"Current local date and time",
   "parameters":{"type":"object","properties":{}}}},
 {"type":"function","function":{"name":"get_weather","description":"Current weather or forecast for a place",
   "parameters":{"type":"object","properties":{"location":{"type":"string"},"when":{"type":"string"}},
                 "required":["location"]}}},
 {"type":"function","function":{"name":"get_radio_status","description":"Live status of the amateur radio hotspot and reflectors",
   "parameters":{"type":"object","properties":{"what":{"type":"string"}}}}},
]

CASES = [
 ("weather",   "What is the weather in Denver right now?",            {"get_weather"}),
 ("weather2",  "What's the forecast for Denver tomorrow at noon?",     {"get_weather"}),
 ("clock",     "What time is it?",                                     {"get_clock"}),
 ("radio",     "Is anyone connected to the P25 reflector right now?",   {"get_radio_status"}),
 ("notool",    "What is a dipole antenna?",                            set()),
]

def ask(prompt, tools, temperature=0.4):
    body = json.dumps({"model":"qwen2.5:3b","stream":False,
        "messages":[{"role":"user","content":prompt}],"tools":tools,
        "options":{"temperature":temperature,"num_predict":80}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

for name, prompt, expected in CASES:
    ok = 0; detail = []
    for i in range(3):
        d = ask(prompt, TOOLS)
        calls = (d.get("message") or {}).get("tool_calls") or []
        got = {c["function"]["name"] for c in calls}
        args = [c["function"].get("arguments") for c in calls]
        ok += (got == expected)
        detail.append(f"{sorted(got) or 'NO-CALL'}{args if args else ''}")
    verdict = "PASS" if ok == 3 else ("PART" if ok else "FAIL")
    print(f"{verdict:4s} {name:9s} expect={sorted(expected) or ['(none)']}  {ok}/3")
    for line in detail: print("        ", line[:110])
