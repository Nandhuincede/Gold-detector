import logging
from langgraph.graph import StateGraph, END
from schemas import Agent_State
from nodes import load_image_node, detect_items_node, save_to_db_node

logger = logging.getLogger(__name__)


NODE_LOAD_IMAGE  = "load_image"
NODE_DETECT      = "detection"
NODE_SAVE_TO_DB  = "store_to_db"


def should_continue_after_load(state: Agent_State) -> str:
    """
    Conditional edge after load_image node.
    If an error occurred, route to END; otherwise continue to detection.
    """
    if state.error:
        logger.warning(f"[graph] Error after load_image: {state.error}. Routing to END.")
        return END
    return NODE_DETECT


def should_continue_after_detect(state: Agent_State) -> str:
    """
    Conditional edge after detection node.
    If an error occurred, route to END; otherwise continue to database save.
    """
    if state.error:
        logger.warning(f"[graph] Error after detection: {state.error}. Routing to END.")
        return END
    return NODE_SAVE_TO_DB


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for ornament detection.

    Pipeline:
        load_image  ──► detection ──► store_to_db ──► END
                   (err)↓         (err)↓
                        END           END

    Returns:
        Compiled LangGraph app ready for .invoke()
    """
    logger.info("[graph] Building LangGraph workflow...")

    
    workflow = StateGraph(Agent_State)

    
    workflow.add_node(NODE_LOAD_IMAGE, load_image_node)
    workflow.add_node(NODE_DETECT,     detect_items_node)
    workflow.add_node(NODE_SAVE_TO_DB, save_to_db_node)

    
    workflow.set_entry_point(NODE_LOAD_IMAGE)

    
    workflow.add_conditional_edges(
        NODE_LOAD_IMAGE,
        should_continue_after_load,
        {END: END, NODE_DETECT: NODE_DETECT},
    )
    workflow.add_conditional_edges(
        NODE_DETECT,
        should_continue_after_detect,
        {END: END, NODE_SAVE_TO_DB: NODE_SAVE_TO_DB},
    )

    
    workflow.add_edge(NODE_SAVE_TO_DB, END)

    
    
    logger.info("[graph] LangGraph workflow compiled successfully.")
    return workflow.compile()



ornament_graph = build_graph()
