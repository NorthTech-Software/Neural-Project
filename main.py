"""
The Neural Project - Interactive Learning Interface
A conversational AI that learns and remembers like a brain
Multilingual support included
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
from datetime import datetime
import json
import os
import subprocess
import tempfile

from neural_core.network import NeuralNetwork
from neural_core.memory import MemorySystem
from adapters import DataHandler

# Multilingual support
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("Note: googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")


class BrainAI:
    """AI brain that learns from conversations"""
    
    def __init__(self):
        self.network = None
        self.memory = MemorySystem()
        self.conversation_history = []
        self.learned_concepts = {}
        self.training_data_X = []
        self.training_data_y = []
        self.concept_embeddings = {}
        self.concept_counter = 0
        self.knowledge_base = {}
        self.code_modules = {}  # Store code from different languages
        
        # Multilingual support
        self.translator = Translator() if TRANSLATION_AVAILABLE else None
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'zh-cn': 'Chinese (Simplified)',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'ru': 'Russian',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        
        # Initialize with basic network
        self._initialize_network()
    
    def _initialize_network(self):
        """Create initial neural network"""
        self.network = NeuralNetwork(
            input_size=128,  # For embedding vectors
            hidden_layers=[64, 32],
            output_size=10,  # Number of concept categories
            activation='relu'
        )
    
    def execute_code(self, code: str, language: str = 'python') -> str:
        """Execute code in specialized programming languages
        
        Languages and purposes:
        - python: Self-changing rules, dynamic behavior modification
        - lisp: Self-changing rules, meta-programming
        - c/assembly: Low-level signals, memory manipulation, hardware control
        - cuda/erlang: Massive parallel computation, distributed processing
        - javascript: Event-driven systems, asynchronous operations
        """
        try:
            if language == 'python':
                return self._execute_python(code)
            elif language == 'lisp':
                return self._execute_lisp(code)
            elif language == 'javascript':
                return self._execute_javascript(code)
            elif language == 'c':
                return self._execute_c_lowlevel(code)
            elif language == 'assembly':
                return self._execute_assembly_lowlevel(code)
            elif language == 'cuda':
                return self._execute_cuda_parallel(code)
            elif language == 'erlang':
                return self._execute_erlang_parallel(code)
            else:
                return f"Language '{language}' not supported"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _execute_python(self, code: str) -> str:
        """Execute Python code for self-changing rules and dynamic behavior"""
        try:
            # Create a controlled environment for rule modification
            local_vars = {'brain': self, 'modify_rule': self._modify_brain_rule}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            return str(local_vars.get('result', 'Rules updated'))
        except Exception as e:
            return f"Python error: {str(e)}"
    
    def _modify_brain_rule(self, rule_name: str, rule_logic: str) -> bool:
        """Modify brain rules dynamically"""
        try:
            # Store rule modifications
            if '_meta_dynamic_rules' not in self.knowledge_base:
                self.knowledge_base['_meta_dynamic_rules'] = {}
            self.knowledge_base['_meta_dynamic_rules'][rule_name] = rule_logic
            return True
        except:
            return False
    
    def _apply_dynamic_learning_rules(self, statement: str):
        """Apply dynamic learning rules (Python self-modification)"""
        if '_meta_dynamic_rules' not in self.knowledge_base:
            return
        
        rules = self.knowledge_base['_meta_dynamic_rules']
        for rule_name, rule_logic in rules.items():
            try:
                local_vars = {
                    'statement': statement,
                    'knowledge_count': len(self.knowledge_base),
                    'brain': self
                }
                exec(rule_logic, {'__builtins__': __builtins__}, local_vars)
            except:
                pass
    
    def _trigger_learning_event(self, event_name: str, event_data: dict):
        """Event-driven learning (JavaScript-inspired)"""
        if not hasattr(self, 'event_handlers'):
            self.event_handlers = {}
        
        # Log event
        self.memory.episodic.record_episode({
            'type': 'event',
            'event_name': event_name,
            'event_data': event_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Execute handlers
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(event_data)
                except:
                    pass
        
        # Auto-consolidation trigger
        if event_name == 'knowledge_added' and len(self.knowledge_base) % 10 == 0:
            self._trigger_learning_event('consolidation_threshold', 
                                        {'count': len(self.knowledge_base)})
    
    def register_event_handler(self, event_name: str, handler_func):
        """Register event handler for reactive learning"""
        if not hasattr(self, 'event_handlers'):
            self.event_handlers = {}
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler_func)
    
    def _execute_javascript(self, code: str) -> str:
        """Execute JavaScript code for event-driven systems and async operations"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                # Add event emitter support
                full_code = "const EventEmitter = require('events');\n" + code
                f.write(full_code)
                f.flush()
                result = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=10)
                os.unlink(f.name)
                return result.stdout if result.stdout else result.stderr
        except FileNotFoundError:
            return "Node.js not installed - required for event-driven systems"
        except subprocess.TimeoutExpired:
            return "JavaScript event-driven timeout"
        except Exception as e:
            return f"JavaScript error: {str(e)}"
    
    def _execute_lisp(self, code: str) -> str:
        """Execute Lisp code for meta-programming and self-modification"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lisp', delete=False) as f:
                f.write(code)
                f.flush()
                # Using sbcl (Steel Bank Common Lisp)
                result = subprocess.run(['sbcl', '--load', f.name, '--non-interactive'], 
                                      capture_output=True, text=True, timeout=10)
                os.unlink(f.name)
                return result.stdout if result.stdout else result.stderr
        except FileNotFoundError:
            return "SBCL not installed - install Steel Bank Common Lisp for meta-programming"
        except subprocess.TimeoutExpired:
            return "Lisp execution timeout"
        except Exception as e:
            return f"Lisp error: {str(e)}"
    
    
    def _execute_assembly_lowlevel(self, code: str) -> str:
        """Execute Assembly code for low-level signal handling and hardware control"""
        try:
            asm_file = tempfile.NamedTemporaryFile(mode='w', suffix='.asm', delete=False)
            obj_file = asm_file.name.replace('.asm', '.obj')
            exe_file = asm_file.name.replace('.asm', '.exe' if os.name == 'nt' else '')
            
            asm_file.write(code)
            asm_file.close()
            
            # Assemble (using NASM) with optimization for signal handling
            assemble_result = subprocess.run(['nasm', '-f', 'win64', asm_file.name, '-o', obj_file], 
                                           capture_output=True, text=True, timeout=10)
            
            if assemble_result.returncode != 0:
                os.unlink(asm_file.name)
                return f"Assembly assembly error: {assemble_result.stderr}"
            
            # Link and execute
            link_result = subprocess.run(['link', obj_file, '/out:' + exe_file], 
                                        capture_output=True, text=True, timeout=10)
            
            if link_result.returncode == 0:
                result = subprocess.run([exe_file], capture_output=True, text=True, timeout=5)
                os.unlink(asm_file.name)
                os.unlink(obj_file)
                os.unlink(exe_file)
                return result.stdout if result.stdout else "Low-level signal handler executed"
            else:
                os.unlink(asm_file.name)
                os.unlink(obj_file)
                return f"Link error: {link_result.stderr}"
        except FileNotFoundError:
            return "NASM or linker not found - required for low-level signal handling"
        except Exception as e:
            return f"Assembly error: {str(e)}"
    
    def _execute_cuda_parallel(self, code: str) -> str:
        """Execute CUDA code for massive parallel computation on GPUs"""
        try:
            cu_file = tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False)
            exe_file = cu_file.name.replace('.cu', '.exe' if os.name == 'nt' else '')
            
            # Add parallel computing headers
            full_code = '#include <stdio.h>\n#include <cuda_runtime.h>\n' + code
            cu_file.write(full_code)
            cu_file.close()
            
            # Compile with parallel optimization
            compile_result = subprocess.run(['nvcc', '-O3', '-arch=sm_70', cu_file.name, '-o', exe_file], 
                                          capture_output=True, text=True, timeout=30)
            
            if compile_result.returncode != 0:
                os.unlink(cu_file.name)
                return f"CUDA parallel compilation error: {compile_result.stderr}"
            
            result = subprocess.run([exe_file], capture_output=True, text=True, timeout=20)
            os.unlink(cu_file.name)
            os.unlink(exe_file)
            
            return result.stdout if result.stdout else result.stderr
        except FileNotFoundError:
            return "NVCC not installed - install CUDA Toolkit for GPU parallel computing"
        except subprocess.TimeoutExpired:
            return "CUDA parallel computation timeout"
        except Exception as e:
            return f"CUDA error: {str(e)}"
    
    def _execute_erlang_parallel(self, code: str) -> str:
        """Execute Erlang code for massive parallel and distributed computing"""
        try:
            erl_file = tempfile.NamedTemporaryFile(mode='w', suffix='.erl', delete=False)
            
            # Add parallel/distributed headers
            full_code = '-module(parallel).\n-compile(export_all).\n' + code
            erl_file.write(full_code)
            erl_file.close()
            
            # Compile and execute with parallel optimization
            result = subprocess.run(['erl', '+P', '1000000', '-noshell', 
                                   '-s', 'c', 'c', erl_file.name,
                                   '-s', 'init', 'stop'], 
                                  capture_output=True, text=True, timeout=30)
            
            os.unlink(erl_file.name)
            return result.stdout if result.stdout else result.stderr
        except FileNotFoundError:
            return "Erlang not installed - install Erlang for distributed parallel computing"
        except subprocess.TimeoutExpired:
            return "Erlang parallel computation timeout"
        except Exception as e:
            return f"Erlang error: {str(e)}"
    
    def _execute_c_lowlevel(self, code: str) -> str:
        """Execute C code for low-level signals and memory manipulation"""
        try:
            c_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False)
            exe_file = c_file.name.replace('.c', '.exe' if os.name == 'nt' else '')
            
            # Prepend signal handling headers
            full_code = '#include <signal.h>\n#include <stdint.h>\n#include <string.h>\n' + code
            c_file.write(full_code)
            c_file.close()
            
            # Compile with optimization for signal handling
            compile_result = subprocess.run(['gcc', '-O2', '-w', c_file.name, '-o', exe_file], 
                                          capture_output=True, text=True, timeout=10)
            
            if compile_result.returncode != 0:
                os.unlink(c_file.name)
                return f"Low-level compilation error: {compile_result.stderr}"
            
            result = subprocess.run([exe_file], capture_output=True, text=True, timeout=5)
            os.unlink(c_file.name)
            os.unlink(exe_file)
            
            return result.stdout if result.stdout else result.stderr
        except FileNotFoundError:
            return "GCC not installed"
        except Exception as e:
            return f"Low-level C error: {str(e)}"
    
    # ==================== MULTILINGUAL SUPPORT ====================
    
    def detect_language(self, text: str) -> dict:
        """Detect the language of input text"""
        if not TRANSLATION_AVAILABLE or not self.translator:
            return {'language': 'en', 'confidence': 1.0, 'name': 'English'}
        
        try:
            detection = self.translator.detect(text)
            lang_code = detection.lang
            confidence = detection.confidence
            lang_name = self.supported_languages.get(lang_code, 'Unknown')
            
            return {
                'language': lang_code,
                'confidence': confidence,
                'name': lang_name
            }
        except Exception as e:
            return {'language': 'en', 'confidence': 0.0, 'name': 'English', 'error': str(e)}
    
    def translate_text(self, text: str, target_lang: str = 'en', source_lang: str = 'auto') -> str:
        """Translate text to target language"""
        if not TRANSLATION_AVAILABLE or not self.translator:
            return text  # Return original if translation unavailable
        
        try:
            if source_lang == 'auto':
                translation = self.translator.translate(text, dest=target_lang)
            else:
                translation = self.translator.translate(text, src=source_lang, dest=target_lang)
            return translation.text
        except Exception as e:
            return f"[Translation error: {str(e)}]"
    
    def process_multilingual_input(self, text: str) -> tuple:
        """Process input in any language and return (english_text, original_language)"""
        # Detect language
        detection = self.detect_language(text)
        original_lang = detection['language']
        
        # Translate to English for processing if needed
        if original_lang != 'en':
            english_text = self.translate_text(text, target_lang='en', source_lang=original_lang)
        else:
            english_text = text
        
        return english_text, original_lang
    
    def respond_in_language(self, response: str, target_lang: str) -> str:
        """Translate response back to user's language"""
        if target_lang == 'en':
            return response
        
        return self.translate_text(response, target_lang=target_lang, source_lang='en')
    
    # ==================== END MULTILINGUAL SUPPORT ====================
    
    def add_knowledge(self, statement: str, auto_expand: bool = False):
        """Learn with dynamic optimization (Python self-modification) and multilingual support"""
        # Process multilingual input
        english_statement, original_lang = self.process_multilingual_input(statement)
        
        # Apply dynamic learning rules if they exist
        if '_meta_dynamic_rules' in self.knowledge_base:
            self._apply_dynamic_learning_rules(english_statement)
        
        # Store in episodic memory with language info
        episode_id = self.memory.episodic.record_episode({
            'statement': english_statement,
            'original_statement': statement,
            'language': original_lang,
            'timestamp': datetime.now().isoformat(),
            'type': 'learning'
        })
        
        # Create variations of the statement for better understanding
        variations = self._generate_variations(english_statement)
        
        # Store original statement
        concept_key = f"concept_{len(self.knowledge_base)}"
        self.knowledge_base[concept_key] = {
            'statement': english_statement,
            'original_statement': statement,
            'language': original_lang,
            'variations': variations,
            'learned_at': datetime.now().isoformat(),
            'reinforcement_count': 1
        }
        
        # Create embedding for original statement
        embedding = self._create_embedding(statement)
        self.concept_embeddings[concept_key] = embedding
        self.memory.long_term.store_pattern(concept_key, embedding)
        
        # Trigger learning events (JavaScript-inspired reactive system)
        self._trigger_learning_event('knowledge_added', {
            'concept': concept_key,
            'statement': statement,
            'embedding_norm': float(np.linalg.norm(embedding))
        })
        
        # Store variations as well (helps with matching)
        for i, variation in enumerate(variations):
            var_key = f"{concept_key}_var{i}"
            var_embedding = self._create_embedding(variation)
            self.concept_embeddings[var_key] = var_embedding
            self.knowledge_base[var_key] = {
                'statement': variation,
                'original_concept': concept_key,
                'learned_at': datetime.now().isoformat(),
                'reinforcement_count': 1,
                'is_variation': True
            }
        
        return episode_id, concept_key
    
    def _generate_variations(self, statement: str) -> list:
        """Generate alternative phrasings of a statement"""
        variations = []
        statement_lower = statement.lower()
        
        # Pattern: "The X is Y" -> "X is Y", "The current X is Y"
        if statement_lower.startswith("the "):
            without_the = statement[4:]
            variations.append(without_the)
            
            # Add "current" for temporal statements
            words = statement.split()
            if len(words) >= 3:
                variations.append(f"The current {without_the}")
                variations.append(f"Current {without_the}")
        
        # Pattern: "X is Y" -> "The X is Y"
        if not statement_lower.startswith("the ") and " is " in statement_lower:
            variations.append(f"The {statement}")
        
        # Pattern: "Next X is Y" -> "X will be Y"
        if statement_lower.startswith("next "):
            rest = statement[5:]
            if " is " in rest:
                parts = rest.split(" is ", 1)
                variations.append(f"{parts[0]} will be {parts[1]}")
                variations.append(f"The next {rest}")
        
        # Pattern for years: "The year is 2025" -> "2025 is the year", "It is 2025"
        if "year is" in statement_lower and any(c.isdigit() for c in statement):
            # Extract the year
            import re
            years = re.findall(r'\b\d{4}\b', statement)
            if years:
                year = years[0]
                variations.append(f"{year} is the year")
                variations.append(f"It is {year}")
                variations.append(f"The current year is {year}")
                variations.append(f"We are in {year}")
        
        # Remove duplicates and the original
        variations = list(set(variations))
        if statement in variations:
            variations.remove(statement)
        
        return variations[:5]  # Limit to 5 variations
    
    def recall_knowledge(self, query: str, response_lang: str = None) -> list:
        """Recall with parallel distributed search (Erlang-inspired) and multilingual support"""
        # Process multilingual query
        english_query, detected_lang = self.process_multilingual_input(query)
        
        # Use detected language for response if not specified
        if response_lang is None:
            response_lang = detected_lang
        
        results = []
        query_embedding = self._create_embedding(english_query)
        
        # Use parallel processing for large knowledge bases (distributed search)
        if len(self.concept_embeddings) > 50:
            try:
                parallel_results = self._parallel_recall(english_query, query_embedding)
                # Translate results if needed
                if response_lang != 'en':
                    for result in parallel_results:
                        result['statement'] = self.respond_in_language(result['statement'], response_lang)
                return parallel_results
            except:
                pass  # Fall back to sequential
        
        # Extract key words/concepts from query
        query_keywords = self._extract_keywords(english_query)
        
        # Find matches in multiple ways
        matches_with_scores = []
        
        for concept_key, embedding in self.concept_embeddings.items():
            statement = self.knowledge_base[concept_key]['statement']
            match_details = {'match_types': []}
            scores = []
            
            statement_lower = statement.lower()
            query_lower = query.lower()
            
            # 1. Check for exact substrings that match the query intent
            exact_substring_score = 0.0
            
            # Check if statement contains key numbers or specific facts from query
            query_words = query.split()
            statement_words = statement.split()
            
            # Look for key nouns (longer words are usually more important)
            important_query_words = [w.lower().strip('.,?!:;') for w in query_words if len(w) > 3]
            important_statement_words = [w.lower().strip('.,?!:;') for w in statement_words if len(w) > 3]
            
            # Count how many important words match
            important_matches = [w for w in important_query_words if w in important_statement_words]
            if important_matches:
                importance_ratio = len(important_matches) / max(len(important_query_words), 1)
                
                # Bonus for having matching important words in similar positions
                query_positions = [i for i, w in enumerate(important_query_words) if w in important_matches]
                stmt_positions = [important_statement_words.index(w) for w in important_matches if w in important_statement_words]
                
                if query_positions and stmt_positions:
                    # Check position correlation (do matching words appear in similar order?)
                    position_match = sum(1 for qp, sp in zip(query_positions, stmt_positions) if abs(qp - sp) < 2)
                    position_score = position_match / max(len(important_matches), 1)
                    
                    if position_score > 0.5:  # Words in similar order
                        exact_substring_score = importance_ratio * 0.9
                    else:
                        exact_substring_score = importance_ratio * 0.6
                else:
                    exact_substring_score = importance_ratio * 0.7
                
                scores.append(('exact', exact_substring_score))
                match_details['match_types'].append('exact-match')
            
            # 2. Keyword-based matching (simpler approach)
            matched_keywords = [kw for kw in query_keywords if kw in statement_lower]
            if matched_keywords and not scores:  # Only if no exact match found
                keyword_ratio = len(matched_keywords) / max(len(query_keywords), 1)
                keyword_score = keyword_ratio * 0.7
                scores.append(('keywords', keyword_score))
                match_details['match_types'].append('keywords')
                match_details['matched_words'] = matched_keywords
            
            # 3. Direct semantic similarity (only fallback)
            similarity = self._cosine_similarity(query_embedding, embedding)
            if not scores and similarity > 0.3:
                scores.append(('semantic', similarity))
                match_details['match_types'].append('semantic')
            
            # Calculate final score
            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                _, final_score = scores[0]
                
                matches_with_scores.append({
                    'concept': concept_key,
                    'statement': statement,
                    'similarity': final_score,
                    'match_type': ', '.join(set(match_details['match_types'])),
                    'matched_words': match_details.get('matched_words', []),
                    'matched_phrase': match_details.get('matched_phrase', '')
                })
        
        # Sort by final score
        matches_with_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Translate results if needed
        top_matches = matches_with_scores[:10]  # Get top 10 matches
        if response_lang != 'en':
            for match in top_matches:
                match['statement'] = self.respond_in_language(match['statement'], response_lang)
        
        return top_matches
    
    def _parallel_recall(self, query: str, query_embedding: np.ndarray) -> list:
        """Parallel distributed search (Erlang-inspired actor model)"""
        # Divide knowledge base into chunks for parallel processing
        chunk_size = max(10, len(self.concept_embeddings) // 4)
        concepts_list = list(self.concept_embeddings.items())
        chunks = [concepts_list[i:i+chunk_size] for i in range(0, len(concepts_list), chunk_size)]
        
        # Process chunks in parallel (simulated)
        all_matches = []
        for chunk in chunks:
            chunk_matches = self._process_chunk(query, query_embedding, chunk)
            all_matches.extend(chunk_matches)
        
        # Merge results
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)
        return all_matches[:10]
    
    def _process_chunk(self, query: str, query_embedding: np.ndarray, chunk: list) -> list:
        """Process chunk of concepts (parallel worker)"""
        query_keywords = self._extract_keywords(query)
        matches = []
        
        for concept_key, embedding in chunk:
            statement = self.knowledge_base[concept_key]['statement']
            score = self._fast_match_score(query, statement, query_embedding, embedding, query_keywords)
            
            if score > 0.3:
                matches.append({
                    'concept': concept_key,
                    'statement': statement,
                    'similarity': score,
                    'match_type': 'parallel-search'
                })
        
        return matches
    
    def _fast_match_score(self, query: str, statement: str, 
                          query_emb: np.ndarray, stmt_emb: np.ndarray,
                          query_kw: list) -> float:
        """Fast match scoring for parallel processing"""
        # Word overlap (fast)
        query_words = set(query.lower().split())
        stmt_words = set(statement.lower().split())
        overlap = len(query_words & stmt_words) / max(len(query_words), 1)
        
        # Keyword match
        kw_match = sum(1 for kw in query_kw if kw in statement.lower()) / max(len(query_kw), 1)
        
        # Semantic (expensive, only if needed)
        if overlap < 0.3 and kw_match < 0.3:
            semantic = self._cosine_similarity(query_emb, stmt_emb)
            return max(overlap * 0.9, kw_match * 0.8, semantic * 0.6)
        
        return max(overlap * 0.9, kw_match * 0.8)
    
    def _extract_keywords(self, text: str) -> list:
        """Extract important keywords from text"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'could', 'should', 'may', 'might', 'can', 'what', 
                     'who', 'where', 'when', 'why', 'how', 'and', 'or', 'but', 'in', 'of', 'to',
                     'that', 'this', 'it', 'for', 'with', 'by', 'on', 'at', 'from'}
        
        words = text.lower().split()
        keywords = [w.strip('.,?!;:') for w in words 
                   if w.strip('.,?!;:') not in stop_words and len(w) > 2]
        return keywords
    
    def _extract_phrases(self, text: str) -> list:
        """Extract important phrases from text"""
        # Remove common question words at start
        text_clean = text.lower()
        for q_word in ['what', 'which', 'who', 'where', 'when', 'why', 'how', 'is', 'are']:
            if text_clean.startswith(q_word):
                text_clean = text_clean[len(q_word):].strip()
                break
        
        # Extract 2-3 word phrases
        words = text_clean.split()
        phrases = []
        
        # Single important words (2+ chars)
        for w in words:
            w_clean = w.strip('.,?!;:')
            if len(w_clean) > 2:
                phrases.append(w_clean)
        
        # 2-word combinations
        for i in range(len(words) - 1):
            phrase = f"{words[i].strip('.,?!;:')} {words[i+1].strip('.,?!;:')}"
            if all(len(w.strip('.,?!;:')) > 2 for w in [words[i], words[i+1]]):
                phrases.append(phrase)
        
        return phrases
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding with enhanced algorithm (C-accelerated for large texts)"""
        # Try C-accelerated version for large texts (low-level signal processing)
        if len(text) > 100:
            try:
                return self._create_embedding_c_accelerated(text)
            except:
                pass  # Fall back to Python
        
        # Enhanced Python: multi-level encoding
        vector = np.zeros(128)
        text_lower = text.lower()
        
        # Level 1: Character encoding (first 64 dims)
        for i, char in enumerate(text_lower[:64]):
            vector[i] = ord(char) / 256.0
        
        # Level 2: Word-level features (dims 64-128)
        words = text_lower.split()[:16]
        for i, word in enumerate(words):
            word_hash = sum(ord(c) for c in word) % 256
            vector[64 + i*4] = word_hash / 256.0
            vector[64 + i*4 + 1] = len(word) / 20.0
            if i < len(words) - 1:
                vector[64 + i*4 + 2] = 1.0  # Word boundary
        
        return vector[:128]
    
    def _create_embedding_c_accelerated(self, text: str) -> np.ndarray:
        """C-accelerated embedding for fast signal-level processing"""
        # Clean text for C
        text_clean = text.replace('"', '\\"').replace('\n', ' ')[:500]
        
        c_code = f'''#include <stdio.h>
#include <string.h>
#include <stdint.h>

int main() {{
    const char* text = "{text_clean}";
    double vector[128];
    memset(vector, 0, sizeof(vector));
    
    int len = strlen(text);
    for(int i = 0; i < len && i < 128; i++) {{
        vector[i] = ((double)(unsigned char)text[i]) / 256.0;
    }}
    
    for(int i = 0; i < 128; i++) {{
        printf("%f ", vector[i]);
    }}
    return 0;
}}'''
        
        result = self.execute_code(c_code, 'c')
        if result and not result.startswith('Error') and not result.startswith('Low-level'):
            try:
                values = [float(x) for x in result.split()]
                if len(values) >= 128:
                    return np.array(values[:128])
            except:
                pass
        raise Exception("C acceleration unavailable")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _check_calendar_knowledge(self, query: str) -> str:
        """Check if query is asking about current date/time - built-in knowledge"""
        query_lower = query.lower()
        now = datetime.now()
        
        # Date-related questions
        if any(word in query_lower for word in ['today', 'date', 'day']):
            if 'what' in query_lower or 'current' in query_lower:
                day_name = now.strftime('%A')
                date_str = now.strftime('%B %d, %Y')
                return f"Today is {day_name}, {date_str}"
        
        # Day of week
        if 'day of the week' in query_lower or 'day is it' in query_lower:
            return now.strftime('%A')
        
        # Current year
        if any(phrase in query_lower for phrase in ['current year', 'what year', 'which year']):
            return str(now.year)
        
        # Current month
        if any(phrase in query_lower for phrase in ['current month', 'what month', 'which month']):
            return now.strftime('%B')
        
        # Time-related questions
        if any(word in query_lower for word in ['time', 'clock']):
            if 'what' in query_lower or 'current' in query_lower:
                return now.strftime('%I:%M %p')
        
        return None  # No calendar knowledge applies
    
    def _infer_new_knowledge(self) -> list:
        """Automatically infer new knowledge from existing memories"""
        import re
        inferred = []
        
        # Look for patterns we can expand (only from user-taught facts, not inferred ones)
        for concept_key, concept_data in self.knowledge_base.items():
            if concept_data.get('is_variation'):
                continue  # Skip variations
            if concept_data.get('inferred'):
                continue  # Skip already inferred facts to prevent cascading
            
            statement = concept_data['statement']
            statement_lower = statement.lower()
            
            # Pattern: "The year is X" -> can infer "Next year is X+1", "Last year was X-1"
            # Only do this for statements that explicitly say "year is"
            years = re.findall(r'\b(\d{4})\b', statement)
            if years and ('year is' in statement_lower or 'the year' in statement_lower):
                year = int(years[0])
                
                # Check if we already know about next/last year (check more thoroughly)
                next_year_exists = any(
                    (str(year + 1) in kb_data['statement'] and 'year' in kb_data['statement'].lower())
                    for kb_data in self.knowledge_base.values()
                    if not kb_data.get('is_variation')
                )
                last_year_exists = any(
                    (str(year - 1) in kb_data['statement'] and 'year' in kb_data['statement'].lower())
                    for kb_data in self.knowledge_base.values()
                    if not kb_data.get('is_variation')
                )
                
                if not next_year_exists:
                    inferred.append({
                        'statement': f"Next year is {year + 1}",
                        'reasoning': f"Inferred from: {statement}"
                    })
                
                if not last_year_exists:
                    inferred.append({
                        'statement': f"Last year was {year - 1}",
                        'reasoning': f"Inferred from: {statement}"
                    })
            
            # Pattern: Simple arithmetic relationships
            # "X is Y" where Y is a number
            if ' is ' in statement_lower:
                parts = statement.split(' is ', 1)
                if len(parts) == 2:
                    try:
                        value = float(parts[1].strip())
                        # Could infer related facts
                        subject = parts[0].strip()
                        
                        # Example: if "temperature is 20", could infer "temperature is not 0"
                        # For now, keep it simple
                    except ValueError:
                        pass
        
        return inferred
    
    def expand_memories(self) -> list:
        """Expand with meta-learning (Lisp-inspired symbolic processing)"""
        # Standard inference
        inferred_facts = self._infer_new_knowledge()
        
        # Meta-learning: symbolic pattern discovery
        meta_patterns = self._meta_learn_patterns()
        for pattern in meta_patterns:
            inferred_facts.append({
                'statement': pattern,
                'reasoning': 'meta-learning pattern discovery'
            })
        
        added = []
        for fact in inferred_facts:
            # Add the inferred knowledge
            episode_id, concept_key = self.add_knowledge(fact['statement'])
            
            # Mark it as inferred
            self.knowledge_base[concept_key]['inferred'] = True
            self.knowledge_base[concept_key]['reasoning'] = fact['reasoning']
            
            added.append(fact)
        
        return added
    
    def _meta_learn_patterns(self) -> list:
        """Meta-learning: symbolic pattern analysis (Lisp-inspired)"""
        meta_facts = []
        
        if len(self.knowledge_base) > 5:
            # Temporal pattern detection
            temporal_count = sum(1 for k, v in self.knowledge_base.items() 
                               if 'year' in v.get('statement', '').lower())
            if temporal_count > 2:
                meta_facts.append(f"I have learned {temporal_count} temporal facts")
            
            # Symbolic word frequency
            all_words = []
            for k, v in self.knowledge_base.items():
                if not v.get('inferred') and not v.get('is_variation'):
                    all_words.extend(v.get('statement', '').lower().split())
            
            from collections import Counter
            if len(all_words) > 20:
                freq = Counter(all_words)
                common = [w for w, c in freq.most_common(5) 
                         if w not in ['the', 'a', 'is', 'are']]
                if common:
                    meta_facts.append(f"Common topics: {', '.join(common[:3])}")
        
        return meta_facts
    
    def generate_answer_summary(self, query: str, results: list) -> str:
        """Generate a direct answer summary from query results"""
        # First check if it's a calendar question (built-in knowledge)
        calendar_answer = self._check_calendar_knowledge(query)
        if calendar_answer:
            return calendar_answer
        
        if not results:
            return "I don't know the answer to that yet. Please teach me!"
        
        # Get the best match
        best_match = results[0]
        statement = best_match['statement']
        score = best_match['similarity']
        
        # If we have a very confident match (>70%), answer directly
        if score > 0.7:
            # Extract the answer from the statement
            query_lower = query.lower()
            statement_lower = statement.lower()
            
            # For "what is" questions, return the part after "is"
            if "what is" in query_lower or "what's" in query_lower:
                if " is " in statement_lower:
                    parts = statement.split(" is ", 1)
                    if len(parts) == 2:
                        return f"{parts[1].strip()}"
            
            # For "when is" questions
            if "when is" in query_lower or "when's" in query_lower:
                if " is " in statement_lower:
                    parts = statement.split(" is ", 1)
                    if len(parts) == 2:
                        answer = f"{parts[1].strip()}"
                        return self._translate_from_english(answer, query_lang)
            
            # For questions asking about current/what/which
            if any(word in query_lower for word in ["current", "what", "which", "when"]):
                # Try to extract year patterns
                import re
                years = re.findall(r'\b\d{4}\b', statement)
                if years and "year" in query_lower:
                    answer = f"{years[0]}"
                    return self._translate_from_english(answer, query_lang)
                
                # Try to extract numbers
                numbers = re.findall(r'\b\d+\b', statement)
                if numbers:
                    # Return statement as-is if it contains an answer
                    return statement
            
            # Default: return the statement
            return statement
        
        # Medium confidence - give the statement with mild uncertainty
        elif score > 0.5:
            return f"Based on what I know: {statement}"
        
        # Low confidence - express uncertainty
        else:
            return f"I'm not completely sure, but maybe: {statement}"
    
    def get_brain_stats(self) -> dict:
        """Get statistics about the brain"""
        return {
            'total_memories': len(self.memory.episodic.episodes),
            'learned_concepts': len(self.knowledge_base),
            'network_neurons': self.network.get_network_info()['total_neurons'],
            'conversations': len(self.conversation_history),
            'network_growth': self.network.get_network_info()['growth_events']
        }


class NeuralBrainGUI:
    """Interactive GUI for teaching the Neural Brain"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Project")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")
        
        # Setup APPDATA path for brain saves
        self.appdata_path = os.path.join(os.getenv('APPDATA'), 'NeuralBrain')
        os.makedirs(self.appdata_path, exist_ok=True)
        self.brain_save_file = os.path.join(self.appdata_path, 'brain_save.json')
        
        self.brain = BrainAI()
        self.teaching = False
        
        self._create_widgets()
        self._update_brain_status()
    
    def _create_widgets(self):
        """Create GUI widgets - simplified"""
        # Header
        header = tk.Frame(self.root, bg="#2d2d2d", height=50)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🧠 Neural Project", 
                        font=("Arial", 14, "bold"), bg="#2d2d2d", fg="#00ff00")
        title.pack(pady=8)
        
        # Main content - single vertical layout
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat/Conversation display
        chat_label = ttk.Label(content, text="Conversation:", font=("Arial", 10, "bold"))
        chat_label.pack(anchor=tk.W, pady=(0, 3))
        
        self.chat_display = tk.Text(content, height=15, width=100, bg="#0d0d0d", 
                                   fg="#00ff00", font=("Courier", 9))
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)
        
        # Input section - simplified
        input_frame = ttk.Frame(content)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Single input field for both teaching and asking
        input_label = ttk.Label(input_frame, text="Say something or ask a question:", 
                               font=("Arial", 9))
        input_label.pack(anchor=tk.W, pady=(0, 2))
        
        input_box = ttk.Frame(input_frame)
        input_box.pack(fill=tk.X)
        
        self.input_text = tk.Entry(input_box, bg="#1a1a1a", fg="#00ff00", 
                                  font=("Courier", 10))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Return>", lambda e: self._handle_input())
        
        # Buttons
        button_box = ttk.Frame(input_box)
        button_box.pack(side=tk.LEFT)
        
        ttk.Button(button_box, text="Teach", command=self._teach_brain).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_box, text="Ask", command=self._query_brain).pack(side=tk.LEFT, padx=2)
        
        # Status bar at bottom
        status_frame = tk.Frame(content, bg="#2d2d2d", height=40)
        status_frame.pack(fill=tk.X, pady=(10, 0), padx=0)
        status_frame.pack_propagate(False)
        
        self.status_text = tk.Label(status_frame, text="Ready", font=("Courier", 11, "bold"), 
                                     fg="#00ff00", bg="#2d2d2d")
        self.status_text.pack(anchor=tk.W, padx=10, pady=8)
        
        # Footer with controls
        footer = ttk.Frame(self.root)
        footer.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(footer, text="Clear Brain", command=self._clear_brain).pack(side=tk.LEFT, padx=5)
        ttk.Button(footer, text="Save Brain", command=self._save_brain).pack(side=tk.LEFT, padx=5)
        ttk.Button(footer, text="Load Brain", command=self._load_brain).pack(side=tk.LEFT, padx=5)
        ttk.Button(footer, text="Expand Memories", command=self._expand_memories).pack(side=tk.LEFT, padx=5)
    
    def _handle_input(self):
        """Handle input - can be teaching or asking"""
        text = self.input_text.get().strip()
        if text:
            # If it's a question (contains ?), ask it. Otherwise teach
            if '?' in text:
                self.query_text = text  # Temporarily set for query method
                self._query_brain_simplified()
            else:
                self._teach_brain()
    
    def _teach_brain(self):
        """Teach the brain new knowledge"""
        statement = self.input_text.get().strip()
        
        if not statement:
            messagebox.showwarning("Empty", "Please type something to teach")
            return
        
        self.teaching = True
        
        thread = threading.Thread(target=self._teach_thread, args=(statement,))
        thread.daemon = True
        thread.start()
    
    def _teach_thread(self, statement: str):
        """Teaching thread with multilingual support"""
        try:
            # Detect language
            detection = self.brain.detect_language(statement)
            lang_info = f" [{detection['name']}]" if detection['language'] != 'en' else ""
            
            # Add to brain's knowledge
            episode_id, concept_key = self.brain.add_knowledge(statement)
            
            # Get variations
            variations = self.brain.knowledge_base[concept_key].get('variations', [])
            
            # Add to conversation history
            self.brain.conversation_history.append({
                'type': 'teach',
                'content': statement,
                'language': detection['language'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Update chat display
            self._append_chat(f"You{lang_info}: {statement}\n", "#00ff00")
            self._append_chat(f"Brain: Learned!\n", "#ffff00")
            
            if variations:
                for var in variations[:2]:  # Show first 2 variations
                    self._append_chat(f"  • {var}\n", "#ffaa00")
            
            # Automatically try to expand memories
            expanded = self.brain.expand_memories()
            if expanded:
                self._append_chat(f"\n💡 Also figured out:\n", "#00aaff")
                for fact in expanded[:2]:  # Show first 2 inferred facts
                    self._append_chat(f"  • {fact['statement']}\n", "#00aaff")
            
            self._append_chat("\n", "#ffff00")
            
            # Clear input
            self.input_text.delete(0, tk.END)
            
            # Update status
            self._update_brain_status()
            
            self.root.update()
        
        except Exception as e:
            self._append_chat(f"Error: {str(e)}\n", "#ff0000")
        
        finally:
            self.teaching = False
    
    def _query_brain(self):
        """Query the brain's knowledge"""
        query = self.input_text.get().strip()
        
        if not query:
            messagebox.showwarning("Empty", "Please ask something")
            return
        
        self._process_query(query)
        self.input_text.delete(0, tk.END)
    
    def _query_brain_simplified(self):
        """Query version for simplified interface"""
        query = self.query_text.strip() if isinstance(self.query_text, str) else self.input_text.get().strip()
        if query:
            self._process_query(query)
            self.input_text.delete(0, tk.END)
    
    def _process_query(self, query: str):
        """Process a query and display answer with multilingual support"""
        try:
            # Detect query language
            detection = self.brain.detect_language(query)
            lang_info = f" [{detection['name']}]" if detection['language'] != 'en' else ""
            
            # Recall knowledge (with automatic translation)
            results = self.brain.recall_knowledge(query)
            answer = self.brain.generate_answer_summary(query, results)
            
            # Translate answer back to user's language if needed
            if detection['language'] != 'en':
                answer = self.brain.respond_in_language(answer, detection['language'])
            
            self._append_chat(f"You{lang_info}: {query}\n", "#00ff00")
            self._append_chat(f"Brain: {answer}\n\n", "#00aaff")
        
        except Exception as e:
            self._append_chat(f"Error: {str(e)}\n", "#ff0000")
    
    def _expand_memories(self):
        """Manually expand brain's memories"""
        try:
            expanded = self.brain.expand_memories()
            
            if expanded:
                self._append_chat(f"\n💡 Brain expanded its memories:\n", "#00aaff")
                for fact in expanded:
                    self._append_chat(f"  • {fact['statement']}\n", "#00aaff")
                
                self._update_brain_status()
            else:
                self._append_chat(f"\n(No new facts to infer yet)\n\n", "#888888")
        
        except Exception as e:
            self._append_chat(f"Error: {str(e)}\n", "#ff0000")
    
    def _run_javascript(self):
        """Run JavaScript code"""
        code = self.input_text.get().strip()
        if not code:
            messagebox.showwarning("Empty", "Enter JavaScript code")
            return
        
        result = self.brain.execute_code(code, 'javascript')
        self._append_chat(f"You (JS): {code}\n", "#00ff00")
        self._append_chat(f"Result: {result}\n\n", "#ff6600")
        self.input_text.delete(0, tk.END)
    
    def _run_lua(self):
        """Run Lua code"""
        code = self.input_text.get().strip()
        if not code:
            messagebox.showwarning("Empty", "Enter Lua code")
            return
        
        result = self.brain.execute_code(code, 'lua')
        self._append_chat(f"You (Lua): {code}\n", "#00ff00")
        self._append_chat(f"Result: {result}\n\n", "#ff6600")
        self.input_text.delete(0, tk.END)
    
    def _append_chat(self, text: str, color: str = "#00ff00"):
        """Append text to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        start_index = self.chat_display.index(tk.END)
        self.chat_display.insert(tk.END, text)
        end_index = self.chat_display.index(tk.END)
        self.chat_display.tag_add("colored", start_index, end_index)
        self.chat_display.tag_configure("colored", foreground=color)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _update_brain_status(self):
        """Update brain status display"""
        stats = self.brain.get_brain_stats()
        
        status_str = f"Memories: {stats['learned_concepts']} | Neurons: {stats['network_neurons']}"
        self.status_text.config(text=status_str)
    
    def _update_memory_display(self, query: str, answer: str, results: list):
        """Display brain's answer summary"""
        pass  # Simplified interface doesn't use this
    
    def _clear_brain(self):
        """Clear all brain data"""
        if messagebox.askyesno("Clear Brain", "Are you sure? This will erase all memories."):
            self.brain = BrainAI()
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self._update_brain_status()
            messagebox.showinfo("Success", "Brain reset and ready to learn!")
    
    def _save_brain(self):
        """Save brain state to file in APPDATA"""
        try:
            brain_data = {
                'knowledge_base': self.brain.knowledge_base,
                'conversation_history': self.brain.conversation_history,
                'stats': self.brain.get_brain_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.brain_save_file, 'w') as f:
                json.dump(brain_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Brain saved to APPDATA")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save brain: {str(e)}")
    
    def _load_brain(self):
        """Load brain state from file in APPDATA"""
        try:
            if not os.path.exists(self.brain_save_file):
                messagebox.showwarning("Not Found", "No save file found in APPDATA")
                return
            
            with open(self.brain_save_file, 'r') as f:
                brain_data = json.load(f)
            
            self.brain.knowledge_base = brain_data['knowledge_base']
            self.brain.conversation_history = brain_data['conversation_history']
            
            # Rebuild embeddings
            for concept_key, info in self.brain.knowledge_base.items():
                embedding = self.brain._create_embedding(info['statement'])
                self.brain.concept_embeddings[concept_key] = embedding
                self.brain.memory.long_term.store_pattern(concept_key, embedding)
            
            self._update_brain_status()
            self._append_chat("Brain loaded from save file!\n\n", "#00ff00")
            messagebox.showinfo("Success", "Brain loaded successfully!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load brain: {str(e)}")


def main():
    """Launch the interactive brain GUI"""
    root = tk.Tk()
    app = NeuralBrainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
