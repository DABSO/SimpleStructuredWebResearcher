from .graph import get_web_searcher_graph


structured_web_searcher = get_web_searcher_graph()

__all__ = ["structured_web_searcher"]

if __name__ == "__main__":
    # get the graph
    graph = get_web_searcher_graph()
    # draw the graph
    png = graph.get_graph().draw_mermaid_png()
    
    # Save to file 
    output_path = "web_searcher_graph.png"
    with open(output_path, "wb") as f:
        f.write(png)
    print(f"Graph saved to {output_path}")
