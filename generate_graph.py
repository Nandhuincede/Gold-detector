from graph import build_graph

compiled = build_graph()
graph_image = compiled.get_graph().draw_mermaid_png()
with open("graph_image.png", "wb") as f:
    f.write(graph_image)
    print("successfully created the graph and save it as graph_image.py")
