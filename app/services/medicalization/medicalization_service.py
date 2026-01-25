"""
Medicalization Service.
Applies colloquial -> clinical transformations with negation guardrails.
"""
import re
import logging
from typing import List, Tuple, Dict, Any, Set
from app.services.medicalization.medicalization_glossary import load_glossary_mappings, GlossaryEntry

logger = logging.getLogger(__name__)

# --- Negation Logic Constants ---
NEGATION_TRIGGERS = {
    "no", "niega", "negativo", "sin", "nunca", "jamÃ¡s", "jamas", "tampoco", "ningun", "ninguna", "ningÃºn"
}

ADVERSATIVES = {
    "pero", "sin embargo", "mas", "aunque", "excepto", "salvo", "sino"
}

# Delimiters that break scope
DELIMITERS_REGEX = re.compile(r"[.;:\n]")

# Connectors that extend list scope: , y ni o
LIST_CONNECTORS_REGEX = re.compile(r"^(,|y|ni|o)\b", re.IGNORECASE)

class NegationSpan:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def overlaps(self, start: int, end: int) -> bool:
        return max(self.start, start) < min(self.end, end)

def _detect_negation_ranges(text: str) -> List[NegationSpan]:
    """
    Detects ranges of text that are effectively negated.
    Implements greedy list extension but stops at adversatives/delimiters.
    """
    spans = []
    text_lower = text.lower()
    n = len(text)
    
    current_scope_start = -1
    
    i = 0
    while i < n:
        # Check delimiters
        if DELIMITERS_REGEX.match(text, i): # match uses string and pos implicitly if compiled? No, match(string, pos) exists on pattern object
            if current_scope_start != -1:
                spans.append(NegationSpan(current_scope_start, i))
                current_scope_start = -1
            i += 1
            continue

        # Check if we are at a word boundary to consider triggers/adversatives
        is_boundary_start = (i == 0 or not text[i-1].isalnum())
        
        if not is_boundary_start:
             i += 1
             continue

        # Now we are at a word start (or symbol)
        
        # Check triggers (if checking for new scope)
        if current_scope_start == -1:
            match = re.match(r"(\w+)\b", text_lower[i:]) # removed leading \b since we checked it
            if match:
                word = match.group(1)
                if word in NEGATION_TRIGGERS:
                    # found trigger
                    current_scope_start = i
                    i += len(word)
                    continue
                    
            # If not trigger, continue
            # Advancing by word length? No, we might miss something?
            # If we matched a word but it wasn't trigger, we can skip it?
            # Safe to just i+=1 if no match, but slower. 
            # If match, skip word.
            if match:
                i += len(match.group(1))
            else:
                i += 1
            continue
            
        # We ARE in a scope. Check Adversatives.
        is_adversative = False
        remaining = text_lower[i:]
        
        # Check Adversatives
        for adv in ADVERSATIVES:
            # Check if text starts with adv AND ends with boundary
            if remaining.startswith(adv):
                # Check boundary after
                end_idx = len(adv)
                if end_idx < len(remaining) and remaining[end_idx].isalnum():
                     continue # Not a whole word match
                
                is_adversative = True
                i_adv_len = len(adv)
                break
        
        if is_adversative:
            spans.append(NegationSpan(current_scope_start, i))
            current_scope_start = -1
            # Resume scan after adversative?
            # i += i_adv_len ? 
            # Yes.
            # Wait, `i` variable in loop.
            # I need to set `i`.
            # But the loop logic above was `i+=1`.
            # I'll update it.
            # wait, `i_adv_len` variable needs to be accessible.
            pass # fallback to logic below
            
        if is_adversative:
             # Logic handled above? No.
             # I need to restructure the loop slightly to be cleaner.
             # But let's just do it here.
             pass # Already appended.
             # i += len(adv)
             # But I need which adv.
             # Handled in break.
             pass
             
        # Re-implementing correctly:
        if current_scope_start != -1: # Inside scope
             # Check Adversative
             matched_adv_len = 0
             for adv in ADVERSATIVES:
                 if remaining.startswith(adv):
                     # check boundary after
                     if len(remaining) == len(adv) or not remaining[len(adv)].isalnum():
                         matched_adv_len = len(adv)
                         break
             
             if matched_adv_len > 0:
                 # End scope
                 spans.append(NegationSpan(current_scope_start, i))
                 current_scope_start = -1
                 i += matched_adv_len
                 continue
                 
        # If not adversative, check if we need to continue or break on something else?
        # List extension logic?
        # "mareo, vomito" -> "mareo" is negated. "," extends. "vomito" is negated.
        # "mareo" is just content.
        # My logic assumes scope closes ONLY on Delimiter or Adversative.
        # So "mareo" is inside scope automatically.
        # We don't need to check "mareo".
        # We just iterate.
        
        # Advance
        # If we matched a word, advance word.
        match = re.match(r"(\w+)\b", text_lower[i:])
        if match:
            i += len(match.group(1))
        else:
            i += 1
        
    # Close pending scope
    if current_scope_start != -1:
        spans.append(NegationSpan(current_scope_start, n))
        
    return spans

def apply_medicalization(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Applies colloquial -> clinical replacements.
    Returns: (medicalized_text, metrics)
    """
    # 1. Detect Negation
    negation_spans = _detect_negation_ranges(text)
    
    # 2. Load Mappings
    mappings = load_glossary_mappings()
    
    # 3. Apply Replacements
    # We must do this carefully to avoid messing up indices.
    # Strategy: Find ALL candidate matches, filter by priority -> non-overlap -> negation guardrail.
    # Then rebuild string.
    
    candidates = [] # (start, end, replacement, priority)
    
    text_lower = text.lower()
    
    for entry in mappings:
        term = entry.term.lower()
        start = 0
        while True:
            idx = text_lower.find(term, start)
            if idx == -1:
                break
                
            # Valid match? Check word boundaries roughly
            # (Simplification: assume user wants strict substring matching but terms usually imply boundaries if needed)
            # Dart implementation likely does regex or string search.
            # We enforce word boundaries if the term starts/ends with alphanumeric.
            
            # Simple check:
            # is start boundary ok?
            char_before = text[idx-1] if idx > 0 else " "
            char_after = text[idx+len(term)] if idx+len(term) < len(text) else " "
            
            is_boundary_start = not char_before.isalnum()
            is_boundary_end = not char_after.isalnum()
            
            if is_boundary_start and is_boundary_end:
                 candidates.append({
                     "start": idx,
                     "end": idx + len(term),
                     "replacement": entry.replacement,
                     "priority": entry.priority,
                     "term": entry.term
                 })
            
            start = idx + 1
            
    # Sort candidates:
    # 1. Priority DESC
    # 2. Length DESC (greedy)
    # 3. Start Index ASC
    candidates.sort(key=lambda x: (x["priority"], x["end"] - x["start"], -x["start"]), reverse=True)
    
    # Filter overlapping and negated
    accepted = []
    
    # Mask array to track usage
    used_mask = [False] * len(text)
    
    for cand in candidates:
        c_start, c_end = cand["start"], cand["end"]
        
        # Check overlaps with used
        if any(used_mask[k] for k in range(c_start, c_end)):
            continue
            
        # Check if inside negation
        # We accept if it does NOT overlap any negation span? 
        # Or if it COMPLETELY inside?
        # Guardrail: "NO aplicar reemplazos dentro del scope negado."
        # So if it overlaps a negation span, skip.
        is_negated = any(span.overlaps(c_start, c_end) for span in negation_spans)
        
        if is_negated:
            # Skip replacement
            continue
            
        # Accept
        accepted.append(cand)
        for k in range(c_start, c_end):
            used_mask[k] = True
            
    # Rebuild string
    # Sort accepted by start index
    accepted.sort(key=lambda x: x["start"])
    
    result_parts = []
    curr = 0
    for cand in accepted:
        result_parts.append(text[curr:cand["start"]])
        
        # Capitalization handling
        original_slice = text[cand["start"]:cand["end"]]
        replacement = cand["replacement"]
        
        if original_slice and original_slice[0].isupper() and replacement:
            replacement = replacement[0].upper() + replacement[1:]
            
        result_parts.append(replacement)
        curr = cand["end"]
    result_parts.append(text[curr:])
    
    final_text = "".join(result_parts)
    
    return final_text, {
        "replacementsCount": len(accepted),
        "negationSpansCount": len(negation_spans),
        "negatedFindings": [] # Placeholder if needed
    }
