from rdflib import Graph, URIRef, Literal, Namespace

# Define a namespace for your project
EX = Namespace("http://example.org/clearsure/")

# Initialize the RDF graph
graph = Graph()
graph.bind("ex", EX)

def add_triple(subject: str, predicate: str, obj: str):
    """Add a triple to the graph."""
    s = URIRef(EX[subject])
    p = URIRef(EX[predicate])
    o = Literal(obj)
    graph.add((s, p, o))

def get_all_triples():
    """Return all triples currently in the graph."""
    return list(graph)

def save_graph(file_path="clearsure_graph.ttl"):
    """Serialize the graph to a Turtle file."""
    graph.serialize(destination=file_path, format="turtle")

def load_graph(file_path="clearsure_graph.ttl"):
    """Load triples from a Turtle file if it exists."""
    try:
        graph.parse(file_path, format="turtle")
    except FileNotFoundError:
        pass
