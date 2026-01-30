#!/usr/bin/env python3
"""
UNIFIED QUANTUM BRAIN WITH ROTATIONAL SPECTROMETRY
==================================================
Integrates RotationalFractalBrain with QuantumSpectrometryGenerator
into a complete unified system for the QLM and driver.

Features:
- Rotational Fractal Brain (skyscraper with rotatable rooms)
- QuantumSpectrometryGenerator (color-based routing)
- QLM-enhanced spectral processing
- Bidirectional brain-spectrometer communication
- Auto-rotation based on prompt keywords

Author: Titan Quantum Brain System
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

print("\n" + "="*70)
print("UNIFIED QUANTUM BRAIN WITH ROTATIONAL SPECTROMETRY")
print("="*70)

try:
    # Import all components
    print("\n[IMPORTING COMPONENTS]")
    from Titan_Quantum_Brainv2_classes import (
        TitanMengerBrain, TitanTrainer, INPUT_DIM,
        RotationalFractalBrain, RotationalTrainer
    )
    from QLM import HybridQFLM, N_QUBITS
    from quantum_spectrometry_generator import (
        QuantumSpectrometryGenerator,
        ColorVectorRouter,
        SpectrumMemoryBank,
        SpectralFeatureExtractor,
        create_spectral_vector,
        create_spectral_vector_from_text,
        SPECTRAL_DIM
    )
    print("✓ All components imported successfully")
    
    # ============================================================
    # UNIFIED ROTATIONAL SPECTROMETRY BRAIN (COMBINED)
    # ============================================================
    
    class UnifiedRotationalBrain(nn.Module):
        """
        Complete unified system with ROTATIONAL architecture:
        - Text Encoder
        - QLM Quantum Processor
        - Rotational Fractal Brain (skyscraper with rotatable rooms)
        - QuantumSpectrometryGenerator (color-based routing)
        - Fusion Layers
        - Spectrum-to-Brain Adapter (FIXED: uses spec_feature_dim)
        
        This is the properly combined version that unifies:
        - UnifiedSpectrometryBrain + RotationalSpectrometryBrain
        """
        
        def __init__(self, 
                     memory_dim: int = 256,
                     spec_feature_dim: int = 64,
                     num_floors: int = 3,
                     rooms_per_floor: int = 8,
                     enable_quantum: bool = True):
            super().__init__()
            
            print("\n[INIT] Building Unified Rotational Spectrometry Brain...")
            print(f"       Feature dim: {spec_feature_dim}, Floors: {num_floors}, Rooms: {rooms_per_floor}")
            
            # Component 1: Text Encoder
            self.text_encoder = nn.Sequential(
                nn.Embedding(256, 128),
                nn.Linear(128, 256),
                nn.ReLU()
            )
            print("  ✓ Text Encoder initialized")
            
            # Component 2: QLM Quantum Processor
            self.qlm = HybridQFLM()
            print("  ✓ QLM Quantum Processor initialized")
            
            # Component 3: Rotational Fractal Brain (NOT TitanMengerBrain!)
            self.brain = RotationalFractalBrain(
                num_floors=num_floors,
                rooms_per_floor=rooms_per_floor,
                enable_quantum=enable_quantum
            )
            print("  ✓ Rotational Fractal Brain initialized")
            
            # Component 4: Spectrometry Generator
            self.spectrometry = QuantumSpectrometryGenerator(
                memory_dim=memory_dim,
                feature_dim=spec_feature_dim,
                enable_quantum=enable_quantum
            )
            print("  ✓ Spectrometry Generator initialized")
            
            # Component 5: Fusion Layers
            self.fusion = nn.Sequential(
                nn.Linear(512 + spec_feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, INPUT_DIM)
            )
            print("  ✓ Fusion layers initialized")
            
            # Component 6: Spectrum-to-Brain Adapter (FIXED: uses spec_feature_dim, not SPECTRAL_DIM)
            # The spectroscopy output is [feature_dim=64], but brain expects [INPUT_DIM=8]
            self.spectrum_to_brain = nn.Linear(spec_feature_dim, INPUT_DIM)
            self.brain_to_spectrum = nn.Linear(INPUT_DIM, SPECTRAL_DIM)
            print("  ✓ Spectrum-Brain adapters initialized")
            
            # Component 7: Output Processor
            self.output_processor = nn.Sequential(
                nn.Linear(INPUT_DIM * 2, INPUT_DIM),
                nn.LayerNorm(INPUT_DIM),
                nn.Tanh()
            )
            print("  ✓ Output processor initialized")
            
            print("\n[INIT] Unified Rotational Brain ready!")
            
        # --- ROTATION METHODS ---
        
        def rotate_prompt(self, prompt: str):
            """Auto-rotate brain based on prompt characteristics."""
            self.brain.rotate_prompt(prompt)
            
        def get_rotation_state(self) -> Dict:
            """Get current rotation state of the brain"""
            return self.brain.get_rotation_state()
            
        def reset_rotations(self):
            """Reset all rotations to identity"""
            self.brain.reset_rotations()
            
        # --- FORWARD PASS ---
        
        def forward(self, text_input: str = None,
                   spectrum_input: torch.Tensor = None,
                   brain_input: torch.Tensor = None,
                   apply_rotations: bool = True) -> Dict:
            """Flexible forward pass supporting multiple input types."""
            outputs = {}
            
            if text_input is not None:
                outputs['text'] = self._encode_text(text_input)
                
            if spectrum_input is not None:
                outputs['spectrum'] = self._process_spectrum(spectrum_input)
                
            if brain_input is not None:
                outputs['brain'] = self._process_brain(brain_input, apply_rotations)
                
            if 'spectrum' in outputs and 'text' in outputs:
                fused = torch.cat([outputs['text']['encoded'], 
                                  outputs['spectrum']['features']], dim=-1)
                outputs['fused'] = {
                    'features': self.output_processor(fused),
                    'spectrum_info': outputs['spectrum']['info']
                }
                
            if 'text' in outputs or 'spectrum' in outputs:
                input_for_brain = outputs.get('fused', {}).get('features', 
                          outputs.get('text', outputs.get('spectrum', {})))
                if input_for_brain is not None:
                    if isinstance(input_for_brain, dict):
                        input_for_brain = input_for_brain.get('features', input_for_brain)
                    outputs['final_brain'] = self.brain(input_for_brain, apply_rotations)
                    
            return outputs
            
        def _encode_text(self, text: str) -> Dict:
            """Encode text to vector"""
            char_indices = torch.tensor([ord(c) for c in text], dtype=torch.long)
            char_indices = char_indices[:256] if len(char_indices) > 256 else torch.cat([
                char_indices, 
                torch.zeros(256 - len(char_indices), dtype=torch.long)
            ])
            
            embedded = self.text_encoder[0](char_indices.unsqueeze(0))
            text_vec = self.text_encoder[1](embedded.mean(dim=1))
            text_vec = self.text_encoder[2](text_vec)
            
            if text_vec.shape[1] < 512:
                text_vec = torch.nn.functional.pad(text_vec, (0, 512 - text_vec.shape[1]))
            elif text_vec.shape[1] > 512:
                text_vec = text_vec[:, :512]
                
            return {'raw': text_vec, 'encoded': text_vec}
            
        def _process_spectrum(self, spectrum: torch.Tensor) -> Dict:
            """Process spectral input through spectrometry generator"""
            output, info = self.spectrometry(spectrum)
            return {'features': output, 'info': info}
            
        def _process_brain(self, brain_input: torch.Tensor, apply_rotations: bool) -> Dict:
            """Process input through rotational brain"""
            output = self.brain(brain_input, apply_rotations=apply_rotations)
            return {'features': output}
            
        # --- PUBLIC API FOR DRIVER ---
        
        def text_to_spectrum(self, text: str) -> torch.Tensor:
            """Convert text to spectral representation"""
            spectrum = create_spectral_vector_from_text(text)
            return spectrum
            
        def spectrum_to_brain_input(self, spectrum: torch.Tensor) -> torch.Tensor:
            """Convert spectrum to brain-compatible input"""
            return self.spectrum_to_brain(spectrum)
            
        def brain_output_to_spectrum(self, brain_output: torch.Tensor) -> torch.Tensor:
            """Convert brain output back to spectrum"""
            return self.brain_to_spectrum(brain_output)
            
        def process_with_spectrometry(self, text: str = None,
                                      spectrum: torch.Tensor = None,
                                      apply_rotations: bool = True) -> Dict:
            """Process input with full spectrometry pipeline."""
            if text is not None:
                if apply_rotations:
                    self.rotate_prompt(text)
                spectrum = create_spectral_vector_from_text(text)
                
            if spectrum is None:
                raise ValueError("Either text or spectrum must be provided")
                
            # Get routing decision
            path_weights, routing_info = self.spectrometry.router(spectrum)
            
            # Process through spectrometry
            spec_output, spec_info = self.spectrometry(spectrum)
            
            # Convert to brain input (FIXED: now uses spec_feature_dim)
            brain_input = self.spectrum_to_brain(spec_output)
            
            # Get rotation state
            rotation_state = self.get_rotation_state() if apply_rotations else None
            
            # Process through rotational brain
            brain_output = self.brain(brain_input, apply_rotations=apply_rotations)
            
            return {
                'input_text': text,
                'input_spectrum': spectrum,
                'routing_info': routing_info,
                'path_weights': path_weights,
                'rotation_state': rotation_state,
                'spectrometry_output': spec_output,
                'spectrometry_info': spec_info,
                'brain_output': brain_output,
                'final_output': brain_output
            }
    
    # ============================================================
    # UNIFIED DRIVER (ENHANCED)
    # ============================================================
    
    class UnifiedDriver:
        """
        Enhanced driver for the Unified Rotational Brain.
        """
        
        def __init__(self, unified_brain: UnifiedRotationalBrain = None,
                     num_floors: int = 3, rooms_per_floor: int = 8):
            print("\n[DRIVER] Initializing Unified Driver...")
            
            if unified_brain is not None:
                self.brain = unified_brain
            else:
                self.brain = UnifiedRotationalBrain(
                    num_floors=num_floors,
                    rooms_per_floor=rooms_per_floor
                )
            
            self.memory = self.brain.spectrometry.memory
            self.router = self.brain.spectrometry.router
            
            print("  ✓ Driver initialized")
            print("  ✓ Access to SpectrumMemoryBank ready")
            print("  ✓ Access to ColorVectorRouter ready")
            
        def process_text(self, text: str, mode: str = 'auto') -> Dict:
            """Process text with full rotational pipeline."""
            print(f"\n[DRIVER] Processing: '{text}'")
            
            # Auto-rotate based on prompt
            self.brain.rotate_prompt(text)
            rotation_state = self.brain.get_rotation_state()
            
            print(f"  [ROTATION] Layer Rotations: {rotation_state['layer_rotations']}")
            print(f"  [ROTATION] Total Operations: {rotation_state['total_operations']}")
            
            if mode == 'auto':
                spectrum = self.brain.text_to_spectrum(text)
            else:
                spectrum = create_spectral_vector(128, 128, 128)
                
            # Get spectrum info
            spectrum_np = spectrum.squeeze().numpy()
            print(f"  [SPECTRUM] R={spectrum_np[0]:.0f} G={spectrum_np[1]:.0f} B={spectrum_np[2]:.0f}")
            print(f"            IR={spectrum_np[3]:.0f} UV={spectrum_np[4]:.0f}")
            
            # Process through unified system
            result = self.brain.process_with_spectrometry(spectrum=spectrum)
            
            # Interpret routing
            routing = result['routing_info']
            print(f"\n  [ROUTING DECISION]")
            print(f"    Processing Intensity: {routing['processing_intensity']:.2f}")
            print(f"    Memory Priority: {routing['memory_priority']:.2f}")
            print(f"    Retrieval Mode: {routing['retrieval_mode']}")
            print(f"    Pattern Match: {'Enabled' if routing['pattern_match_enabled'] else 'Disabled'}")
            print(f"    Exact Match: {'Enabled' if routing['exact_match_enabled'] else 'Disabled'}")
            
            # Output interpretation
            output_norm = torch.norm(result['final_output']).item()
            print(f"\n  [OUTPUT]")
            print(f"    Final Output Norm: {output_norm:.4f}")
            
            # Generate response
            response = self._generate_response(result, text)
            
            return {
                'input_text': text,
                'rotation_state': rotation_state,
                'spectrum': spectrum,
                'routing_info': routing,
                'output': result['final_output'],
                'response': response
            }
            
        def manual_rotate(self, layer_idx: int, axis: str, degrees: float):
            """Manually rotate a specific layer"""
            self.brain.brain.engine.rotate_layer(layer_idx, axis, degrees)
            print(f"[DRIVER] Rotated layer {layer_idx} around {axis}-axis by {degrees}°")
            
        def rubik_move(self, face: str, direction: str = 'clockwise'):
            """Perform Rubik's cube style move"""
            self.brain.brain.engine.rubik_move(face, direction)
            print(f"[DRIVER] Rubik's move: {face} - {direction}")
            
        def reset_rotations(self):
            """Reset all rotations to identity"""
            self.brain.reset_rotations()
            print("[DRIVER] All rotations reset to identity")
            
        def _generate_response(self, result: Dict, original_text: str) -> str:
            """Generate text response based on processing result"""
            routing = result['routing_info']
            rotation = result.get('rotation_state', {})
            
            total_rot = sum(rotation.get('layer_rotations', [0]))
            
            if total_rot > 180:
                rot_desc = "heavily rotated"
            elif total_rot > 90:
                rot_desc = "moderately rotated"
            else:
                rot_desc = "lightly rotated"
                
            if routing['processing_intensity'] > 0.7:
                intensity_desc = "high-energy"
            elif routing['processing_intensity'] > 0.4:
                intensity_desc = "moderate-energy"
            else:
                intensity_desc = "low-energy"
                
            if routing['retrieval_mode'] == 'fuzzy':
                mode_desc = "pattern-based"
            else:
                mode_desc = "exact-match"
                
            return (f"Processed '{original_text}' with {intensity_desc} "
                   f"processing ({mode_desc} retrieval, {rot_desc}).")
    
    # ============================================================
    # DEMONSTRATION
    # ============================================================
    
    def run_demo():
        """Run demonstration of the unified rotational system"""
        print("\n" + "="*70)
        print("DEMO: UNIFIED ROTATIONAL SPECTROMETRY BRAIN")
        print("="*70)
        
        # Initialize unified brain
        print("\n[INIT] Creating Unified Rotational Brain...")
        unified_brain = UnifiedRotationalBrain(
            num_floors=3,
            rooms_per_floor=8,
            memory_dim=256,
            spec_feature_dim=64,
            enable_quantum=True
        )
        unified_brain.eval()
        
        # Initialize driver
        print("\n[INIT] Creating Unified Driver...")
        driver = UnifiedDriver(unified_brain)
        
        # Test prompts with auto-rotation
        test_prompts = [
            "Analyze this code structure",
            "Create a new function for me",
            "Find the bug in this logic",
            "Transform the data format",
            "Explain how neural networks work"
        ]
        
        results = []
        
        for text in test_prompts:
            result = driver.process_text(text)
            results.append(result)
            print("\n" + "-"*50)
            
        # Manual rotation demo
        print("\n" + "="*70)
        print("MANUAL ROTATION DEMO")
        print("="*70)
        
        driver.manual_rotate(0, 'z', 90)
        driver.rubik_move('top', 'clockwise')
        driver.reset_rotations()
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nCOMPONENTS DEMONSTRATED:")
        print("  ✓ UnifiedRotationalBrain (combined architecture)")
        print("  ✓ UnifiedDriver")
        print("  ✓ Auto-rotation based on prompts")
        print("  ✓ Manual layer rotation")
        print("  ✓ Rubik's cube moves")
        print("  ✓ Rotation state tracking")
        print("  ✓ Color-directed routing")
        print("  ✓ SpectrumMemoryBank integration")
        
        return results
    
    # ============================================================
    # MAIN EXECUTION
    # ============================================================
    
    if __name__ == "__main__":
        results = run_demo()
        
        print("\n" + "="*70)
        print("UNIFIED ROTATIONAL BRAIN OPERATIONAL")
        print("="*70)
        print("\nUSAGE:")
        print("  1. Create UnifiedRotationalBrain()")
        print("  2. Use UnifiedDriver for simplified API")
        print("  3. Auto-rotation: brain.rotate_prompt('Analyze this...')")
        print("  4. Manual rotation: brain.brain.engine.rotate_layer(0, 'z', 90)")
        print("\nROTATION PATTERNS:")
        print("  - 'Analyze', 'Review' -> Z-axis rotation")
        print("  - 'Create', 'Build' -> X-axis rotation")
        print("  - 'Find', 'Search' -> Y-axis rotation")
        print("  - 'Transform', 'Convert' -> Combined rotations")
        print("  - 'Explain', 'Describe' -> Gentle rotation")
        print("\n" + "="*70 + "\n")

except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("\nMake sure all required files are present:")
    print("  - Titan_Quantum_Brainv2_classes.py")
    print("  - QLM.py")
    print("  - quantum_spectrometry_generator.py")
    import sys
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

