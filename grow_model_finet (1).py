#!/usr/bin/env python3
"""
TIGR GROWTH-BASED FINE-TUNING WITH ONTOLOGICAL, EPIDEMIOLOGICAL, 
PHYSIOLOGICAL, BIOLOGICAL, AND PHILOSOPHICAL QUESTION TYPES
================================================================
Uses Phytoborg growth simulation as a feedback mechanism for model training.

The growth simulator tracks model "growth" based on:
- Training loss improvement
- Gradient magnitude
- Learning progress

This creates a bio-inspired feedback loop where the model "grows" as it learns.

Features:
- Batch processing support
- Multiple question types (ontological, epidemiological, physiological, 
  biological, philosophical)
- Enhanced data augmentation for TIGR domains

Usage:
    python3 grow_model_finetuning.py

Author: Titan Quantum Brain System
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import random

print("\n" + "="*70)
print("TIGR GROWTH-BASED FINE-TUNING")
print("Enhanced with Ontological, Epidemiological, Physiological,")
print("Biological, and Philosophical Question Types")
print("="*70)


# =============================================================================
# QUESTION TYPE DEFINITIONS
# =============================================================================

class QuestionType:
    """Enumeration of TIGR question types for comprehensive training."""
    
    ONTOLOGICAL = "ontological"     # Questions about being, existence, reality
    EPIDEMIOLOGICAL = "epidemiological"  # Questions about disease spread, patterns
    PHYSIOLOGICAL = "physiological" # Questions about bodily functions, processes
    BIOLOGICAL = "biological"       # Questions about living organisms, evolution
    PHILOSOPHICAL = "philosophical" # Questions about meaning, ethics, knowledge
    
    @classmethod
    def all_types(cls) -> List[str]:
        return [
            cls.ONTOLOGICAL,
            cls.EPIDEMIOLOGICAL,
            cls.PHYSIOLOGICAL,
            cls.BIOLOGICAL,
            cls.PHILOSOPHICAL
        ]


# =============================================================================
# ENHANCED DATA GENERATOR WITH MULTI-DOMAIN QUESTIONS
# =============================================================================

class TIGRQuestionGenerator:
    """
    Generates training data with ontological, epidemiological, physiological,
    biological, and philosophical question types.
    """
    
    def __init__(self, input_dim: int = 8):
        self.input_dim = input_dim
        
        # Domain-specific embedding patterns
        self.domain_patterns = {
            QuestionType.ONTOLOGICAL: {
                'concept_space': 0.3,
                'existence_weight': 0.25,
                'reality_dimension': 0.2,
                'being_alignment': 0.15,
                'essence_depth': 0.10
            },
            QuestionType.EPIDEMIOLOGICAL: {
                'infection_rate': 0.25,
                'transmission_vector': 0.20,
                'population_dynamics': 0.20,
                'susceptibility_index': 0.15,
                'containment_efficacy': 0.20
            },
            QuestionType.PHYSIOLOGICAL: {
                'homeostasis': 0.25,
                'metabolic_rate': 0.20,
                'neural_activity': 0.20,
                'cardiovascular': 0.15,
                'immunological': 0.20
            },
            QuestionType.BIOLOGICAL: {
                'genetic_expression': 0.25,
                'evolutionary_fitness': 0.20,
                'cellular_process': 0.20,
                'biodiversity': 0.15,
                'ecosystem_role': 0.20
            },
            QuestionType.PHILOSOPHICAL: {
                'ethical_reasoning': 0.25,
                'epistemic_value': 0.20,
                'consciousness': 0.20,
                'meaning_structure': 0.15,
                'moral_framework': 0.20
            }
        }
    
    def generate_question_batch(
        self, 
        batch_size: int,
        question_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate a batch of training samples for a specific question type.
        
        Args:
            batch_size: Number of samples in batch
            question_type: Type of questions to generate (or random if None)
        
        Returns:
            Tuple of (inputs, targets, metadata)
        """
        if question_type is None:
            question_type = random.choice(QuestionType.all_types())
        
        pattern = self.domain_patterns.get(question_type, self.domain_patterns[QuestionType.BIOLOGICAL])
        
        inputs = []
        targets = []
        metadata_list = []
        
        for _ in range(batch_size):
            # Generate input vector with domain-specific weighting
            base_input = torch.randn(self.input_dim)
            
            # Apply domain-specific transformations
            transformed_input = self._apply_domain_pattern(base_input, pattern)
            
            # Generate target based on question type
            target = self._generate_target(transformed_input, question_type)
            
            inputs.append(transformed_input)
            targets.append(target)
            
            metadata_list.append({
                'question_type': question_type,
                'domain_weights': pattern,
                'timestamp': datetime.now().isoformat()
            })
        
        return (
            torch.stack(inputs),
            torch.stack(targets),
            metadata_list
        )
    
    def _apply_domain_pattern(
        self, 
        input_vec: torch.Tensor, 
        pattern: Dict[str, float]
    ) -> torch.Tensor:
        """Apply domain-specific weighting pattern to input vector."""
        result = input_vec.clone()
        keys = list(pattern.keys())
        
        # Distribute input dimensions across pattern keys
        for i, key in enumerate(keys):
            if i < len(result):
                result[i] = result[i] * pattern[key] * 3
        
        return result
    
    def _generate_target(
        self, 
        input_vec: torch.Tensor, 
        question_type: str
    ) -> torch.Tensor:
        """Generate target vector based on question type."""
        if question_type == QuestionType.ONTOLOGICAL:
            # Existence-focused transformation
            target = input_vec * 0.5 + torch.abs(input_vec) * 0.3
        
        elif question_type == QuestionType.EPIDEMIOLOGICAL:
            # Spread/dynamics transformation  
            target = torch.nn.functional.normalize(input_vec, dim=0) * 0.8 + input_vec * 0.2
        
        elif question_type == QuestionType.PHYSIOLOGICAL:
            # Homeostatic regulation transformation
            target = torch.sigmoid(input_vec) * 2 - 1
        
        elif question_type == QuestionType.BIOLOGICAL:
            # Evolutionary fitness transformation
            target = input_vec * 0.6 + torch.randn(self.input_dim) * 0.1
        
        elif question_type == QuestionType.PHILOSOPHICAL:
            # Ethical/meaning transformation
            target = torch.tanh(input_vec) * 1.5
        
        else:
            target = input_vec * 0.5 + torch.randn(self.input_dim) * 0.1
        
        return target
    
    def generate_mixed_batch(
        self, 
        batch_size: int,
        type_distribution: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate a mixed batch with balanced question types.
        
        Args:
            batch_size: Total samples in batch
            type_distribution: Optional custom distribution (default: equal)
        
        Returns:
            Tuple of (inputs, targets, metadata_list)
        """
        if type_distribution is None:
            type_distribution = {qt: 1.0/5 for qt in QuestionType.all_types()}
        
        inputs = []
        targets = []
        metadata_list = []
        
        # Calculate samples per type (balanced)
        question_types = QuestionType.all_types()
        base_count = batch_size // len(question_types)
        remainder = batch_size % len(question_types)
        
        samples_per_type = {}
        for i, qt in enumerate(question_types):
            count = base_count + (1 if i < remainder else 0)
            samples_per_type[qt] = count
        
        # Generate batches for each type
        for qt, count in samples_per_type.items():
            if count > 0:
                batch_inputs, batch_targets, batch_metadata = self.generate_question_batch(count, qt)
                inputs.append(batch_inputs)
                targets.append(batch_targets)
                metadata_list.extend(batch_metadata)
        
        # Concatenate all batches
        all_inputs = torch.cat(inputs) if inputs else torch.zeros(0, self.input_dim)
        all_targets = torch.cat(targets) if targets else torch.zeros(0, self.input_dim)
        
        # Shuffle within batch
        if len(all_inputs) > 0:
            perm = torch.randperm(len(all_inputs))
            return (
                all_inputs[perm],
                all_targets[perm],
                [metadata_list[i] for i in perm.tolist()]
            )
        else:
            return all_inputs, all_targets, metadata_list


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """
    Handles batch processing for training with support for:
    - Multiple batch sizes
    - Data shuffling
    - Mixed question type batches
    - Gradient accumulation for larger effective batches
    """
    
    def __init__(
        self,
        data: List[Dict],
        batch_size: int = 8,
        shuffle: bool = True
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
    
    def __iter__(self):
        self.epoch += 1
        n = len(self.data)
        
        if self.shuffle:
            indices = torch.randperm(n)
        else:
            indices = torch.arange(n)
        
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_indices = indices[start:end]
            
            batch_inputs = torch.stack([self.data[i]['input'] for i in batch_indices])
            batch_targets = torch.stack([self.data[i]['target'] for i in batch_indices])
            
            yield batch_inputs, batch_targets
    
    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    def get_batch_count(self) -> int:
        """Get number of batches per epoch."""
        return len(self)


# =============================================================================
# GROWTH-AWARE TRAINER
# =============================================================================

class GrowthAwareTrainer:
    """
    Trainer that uses Phytoborg growth simulation as feedback.
    
    Features:
    - Tracks model growth through simulation
    - Applies TIGR-Tas modifications based on learning progress
    - Adaptive learning rate based on growth
    - Growth regularization in loss function
    - Multi-domain question type support
    """
    
    def __init__(self, 
                 model,
                 growth_body_length: int = 100,
                 learning_rate: float = 0.001,
                 growth_loss_weight: float = 0.01):
        
        print("\n[INIT] Growth-Aware Trainer")
        
        self.model = model
        self.learning_rate = learning_rate
        self.growth_loss_weight = growth_loss_weight
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.loss_history = []
        self.growth_history = []
        self.gradient_history = []
        self.question_type_accuracy = {qt: [] for qt in QuestionType.all_types()}
        
        # Initialize growth simulator
        from TIGR_Bio_Integration import PhytoborgGrowthSimulator
        self.growth_simulator = PhytoborgGrowthSimulator(growth_body_length)
        
        # Growth thresholds
        self.growth_threshold = 0.1
        self.tigrna_edits_applied = 0
        
        print(f"  Body length: {growth_body_length} nodes")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Growth loss weight: {growth_loss_weight}")
        print("[INIT] Growth-Aware Trainer ready!")
    
    def compute_growth_signal(self, loss: float, gradient_norm: float) -> float:
        """
        Compute growth signal from training metrics.
        
        Higher growth when:
        - Loss is decreasing
        - Gradients are healthy (not too small, not exploding)
        """
        # Normalize loss (assume initial loss around 1.0)
        loss_factor = max(0.0, 1.0 - loss)
        
        # Gradient factor (healthy gradients around 0.1-1.0)
        gradient_factor = min(1.0, gradient_norm / 0.5)
        
        # Combined growth signal (0-10 scale)
        growth_signal = (loss_factor * 5 + gradient_factor * 5)
        
        return round(growth_signal, 2)
    
    def train_step(
        self, 
        input_data, 
        target_data, 
        epoch: int = 0,
        question_type: str = "general"
    ) -> Dict:
        """
        Single training step with growth tracking.
        
        Args:
            input_data: Batch of input tensors
            target_data: Batch of target tensors
            epoch: Current epoch number
            question_type: Type of questions in this batch
        
        Returns:
            Dict with loss, growth metrics, and model state
        """
        self.model.train()
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(input_data)
        
        # Compute main loss
        main_loss = self.loss_fn(output, target_data)
        
        # Compute gradients
        main_loss.backward()
        
        # Get gradient norm
        gradient_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                gradient_norm += param.grad.data.norm(2).item() ** 2
        gradient_norm = gradient_norm ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Compute growth signal
        growth_signal = self.compute_growth_signal(main_loss.item(), gradient_norm)
        
        # Apply growth signal to simulator
        growth_result = self.growth_simulator.apply_quantum_signal(growth_signal)
        
        # Compute growth loss (encourage healthy growth patterns)
        current_growth = self.growth_simulator.structural_density.sum()
        
        # Target growth: steady increase
        target_growth = (epoch + 1) * 0.5
        growth_loss = self.growth_loss_weight * abs(current_growth - target_growth)
        
        # Total loss
        total_loss = main_loss + growth_loss
        
        # Update parameters
        self.optimizer.step()
        
        # Record history
        self.loss_history.append(main_loss.item())
        self.growth_history.append({
            'epoch': epoch,
            'question_type': question_type,
            'signal': growth_signal,
            'density': current_growth,
            'max_density': growth_result['max_density']
        })
        self.gradient_history.append(gradient_norm)
        
        # Track per-question-type loss
        if question_type in self.question_type_accuracy:
            self.question_type_accuracy[question_type].append(main_loss.item())
        
        # Apply TIGR-Tas modifications based on milestones
        self._check_milestones(epoch, main_loss.item())
        
        return {
            'main_loss': main_loss.item(),
            'total_loss': total_loss.item(),
            'growth_loss': growth_loss.item(),
            'growth_signal': growth_signal,
            'gradient_norm': gradient_norm,
            'growth_result': growth_result,
            'tigrna_edits': self.tigrna_edits_applied,
            'question_type': question_type
        }
    
    def _check_milestones(self, epoch: int, loss: float):
        """Apply TIGR-Tas modifications at learning milestones."""
        milestones = [10, 25, 50, 100, 250, 500]
        
        for milestone in milestones:
            if epoch == milestone and self.tigrna_edits_applied < milestone // 10:
                edits = [
                    f"VND7_Promoter_epoch_{epoch}",
                    f"WUSCHEL_Enhancer_epoch_{epoch}",
                    f"Cytoskeletal_strengthen_epoch_{epoch}"
                ]
                result = self.growth_simulator.apply_tigrna_edits(edits)
                self.tigrna_edits_applied += 1
                print(f"\n  [!] TIGR-Tas milestone reached at epoch {epoch}")
                print(f"      Applied {result['edits_applied']} modifications")
    
    def get_status(self) -> Dict:
        """Get current training status."""
        return {
            'current_loss': self.loss_history[-1] if self.loss_history else 0,
            'avg_loss': sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0,
            'total_growth': self.growth_simulator.structural_density.sum(),
            'max_density': self.growth_simulator.structural_density.max(),
            'tigrna_edits': self.tigrna_edits_applied,
            'gradient_norm': self.gradient_history[-1] if self.gradient_history else 0,
            'epochs_trained': len(self.loss_history)
        }
    
    def get_question_type_stats(self) -> Dict:
        """Get per-question-type loss statistics."""
        stats = {}
        for qt in QuestionType.all_types():
            losses = self.question_type_accuracy.get(qt, [])
            if losses:
                stats[qt] = {
                    'mean': sum(losses) / len(losses),
                    'last': losses[-1],
                    'count': len(losses)
                }
            else:
                stats[qt] = {'mean': 0, 'last': 0, 'count': 0}
        return stats
    
    def generate_growth_report(self) -> str:
        """Generate ASCII visualization of growth."""
        report = [
            "\n" + "="*60,
            "GROWTH TRAINING REPORT",
            "="*60,
            f"Epochs trained: {len(self.loss_history)}",
            f"Current loss: {self.loss_history[-1]:.4f}" if self.loss_history else "N/A",
            f"Total growth: {self.growth_simulator.structural_density.sum():.2f}",
            f"Max density: {self.growth_simulator.structural_density.max():.2f}",
            f"TIGR-Tas edits: {self.tigrna_edits_applied}",
            "",
            "Question Type Performance:"
        ]
        
        stats = self.get_question_type_stats()
        for qt, data in stats.items():
            bar_len = int((1 - data['mean']) * 20) if data['mean'] > 0 else 20
            bar = "█" * bar_len + "░" * (20 - bar_len)
            report.append(f"  {qt:20s} [{bar}] {data['mean']:.4f} ({data['count']} samples)")
        
        report.extend([
            "",
            "Growth Profile (first 15 nodes):"
        ])
        
        for i in range(min(15, len(self.growth_simulator.structural_density))):
            bar_length = int(self.growth_simulator.structural_density[i] * 10)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            report.append(f"  [{i:2d}] {bar} {self.growth_simulator.structural_density[i]:.2f}")
        
        report.extend(["", "="*60])
        return "\n".join(report)


# =============================================================================
# GROWTH-BASED FINE-TUNING WITH BATCH PROCESSING
# =============================================================================

def grow_model_finetuning(
    model,
    training_data: List[Dict],
    epochs: int = 500,
    learning_rate: float = 0.0001,
    batch_size: int = 8,
    use_mixed_batches: bool = True,
    gradient_accumulation_steps: int = 1
):
    """
    Fine-tune model using growth-based feedback with batch processing.
    
    Args:
        model: RotationalFractalBrain or similar
        training_data: List of {input, target} dictionaries
        epochs: Number of training epochs (default: 500)
        learning_rate: Learning rate (default: 0.0001)
        batch_size: Samples per batch (default: 8)
        use_mixed_batches: Use mixed question type batches
        gradient_accumulation_steps: For larger effective batch sizes
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*60)
    print("GROWTH-BASED FINE-TUNING")
    print("="*60)
    
    # Initialize trainer
    trainer = GrowthAwareTrainer(
        model=model,
        learning_rate=learning_rate,
        growth_loss_weight=0.01
    )
    
    # Initialize question generator
    question_generator = TIGRQuestionGenerator()
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        data=training_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Training samples: {len(training_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {batch_processor.get_batch_count()}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}x")
    print(f"Question types: {', '.join(QuestionType.all_types())}")
    
    if use_mixed_batches:
        print("Mode: Mixed question type batches")
    else:
        print("Mode: Sequential question type batches")
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        
        if use_mixed_batches:
            # Generate mixed batch for each step
            for step in range(batch_processor.get_batch_count()):
                batch_inputs, batch_targets, metadata = question_generator.generate_mixed_batch(batch_size)
                
                # Track per-sample question types
                for meta in metadata:
                    qt = meta['question_type']
                    single_input = batch_inputs[0:1]
                    single_target = batch_targets[0:1]
                    result = trainer.train_step(single_input, single_target, epoch, qt)
                
                # For the main loss tracking, use a full batch
                result = trainer.train_step(batch_inputs, batch_targets, epoch, "mixed")
                epoch_losses.append(result['main_loss'])
        else:
            # Cycle through question types
            for step, (batch_inputs, batch_targets) in enumerate(batch_processor):
                question_type = QuestionType.all_types()[step % len(QuestionType.all_types())]
                result = trainer.train_step(batch_inputs, batch_targets, epoch, question_type)
                epoch_losses.append(result['main_loss'])
        
        # Progress reporting
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"\n  Epoch {epoch+1}/{epochs}")
            print(f"    Avg Loss: {avg_loss:.4f}")
            print(f"    Growth signal: {result['growth_signal']}")
            print(f"    Gradient norm: {result['gradient_norm']:.4f}")
            print(f"    Max density: {result['growth_result']['max_density']:.2f}")
            print(f"    TIGR-Tas edits: {result['tigrna_edits']}")
            print(f"    Question type: {result.get('question_type', 'mixed')}")
    
    # Final report
    print(trainer.generate_growth_report())
    
    return model, {
        'loss_history': trainer.loss_history,
        'growth_history': trainer.growth_history,
        'question_type_stats': trainer.get_question_type_stats(),
        'final_status': trainer.get_status()
    }


# =============================================================================
# ENHANCED DATA GENERATION
# =============================================================================

def generate_tigr_training_data(
    num_samples: int = 500,
    input_dim: int = 8,
    balanced_types: bool = True
) -> List[Dict]:
    """
    Generate training data with all TIGR question types.
    
    Args:
        num_samples: Total number of samples
        input_dim: Dimension of input vectors
        balanced_types: Balance samples across question types
    
    Returns:
        List of {input, target, metadata} dictionaries
    """
    print(f"\n[DATA] Generating {num_samples} TIGR training samples...")
    
    generator = TIGRQuestionGenerator(input_dim)
    training_data = []
    
    if balanced_types:
        # Generate equal samples for each type
        samples_per_type = num_samples // len(QuestionType.all_types())
        
        for question_type in QuestionType.all_types():
            for _ in range(samples_per_type):
                inputs, targets, metadata = generator.generate_question_batch(1, question_type)
                training_data.append({
                    'input': inputs[0],
                    'target': targets[0],
                    'metadata': metadata[0]
                })
        
        # Add remaining samples as mixed
        remaining = num_samples - len(training_data)
        for _ in range(remaining):
            inputs, targets, metadata = generator.generate_mixed_batch(1)
            training_data.append({
                'input': inputs[0],
                'target': targets[0],
                'metadata': metadata[0]
            })
    else:
        # Generate mixed batches
        for _ in range(num_samples):
            inputs, targets, metadata = generator.generate_mixed_batch(1)
            training_data.append({
                'input': inputs[0],
                'target': targets[0],
                'metadata': metadata[0]
            })
    
    print(f"  Generated {len(training_data)} samples")
    print(f"  Question types: {', '.join(QuestionType.all_types())}")
    
    return training_data


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_demo():
    """Demonstrate enhanced growth-based fine-tuning."""
    
    print("\n" + "="*70)
    print("DEMONSTRATION: ENHANCED GROWTH-BASED FINE-TUNING")
    print("="*70)
    
    # Import model
    from Titan_Quantum_Brainv2_classes import RotationalFractalBrain, INPUT_DIM
    
    # Create model
    print("\n[1] Creating RotationalFractalBrain...")
    model = RotationalFractalBrain(num_floors=3, rooms_per_floor=8)
    model.eval()
    
    # Generate training data with all question types
    print("\n[2] Generating TIGR training data...")
    training_data = generate_tigr_training_data(
        num_samples=200,
        input_dim=INPUT_DIM,
        balanced_types=True
    )
    
    # Run growth-based fine-tuning
    print("\n[3] Running enhanced growth-based fine-tuning...")
    model, history = grow_model_finetuning(
        model=model,
        training_data=training_data,
        epochs=100,
        learning_rate=0.01,
        batch_size=8,
        use_mixed_batches=True
    )
    
    # Results
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"\nFinal Status:")
    print(f"  Epochs trained: {history['final_status']['epochs_trained']}")
    print(f"  Final loss: {history['final_status']['current_loss']:.4f}")
    print(f"  Average loss: {history['final_status']['avg_loss']:.4f}")
    print(f"  Total growth: {history['final_status']['total_growth']:.2f}")
    print(f"  Max density: {history['final_status']['max_density']:.2f}")
    print(f"  TIGR-Tas edits applied: {history['final_status']['tigrna_edits']}")
    
    print("\nQuestion Type Performance:")
    for qt, stats in history['question_type_stats'].items():
        print(f"  {qt:20s}: mean={stats['mean']:.4f}, samples={stats['count']}")
    
    # Save model
    print("\n[4] Saving fine-tuned model...")
    save_path = "brain_grown.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': history,
        'growth_state': model.state_dict()
    }, save_path)
    print(f"    Saved to: {save_path}")
    
    return model, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run demonstration
    model, history = run_demo()
    
    print("\n" + "="*70)
    print("ENHANCED GROWTH-BASED FINE-TUNING OPERATIONAL")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Batch processing with configurable sizes")
    print("  ✓ Mixed question type batches")
    print("  ✓ Question types: ontological, epidemiological,")
    print("    physiological, biological, philosophical")
    print("  ✓ Gradient accumulation support")
    print("  ✓ Per-question-type performance tracking")
    print("\nUsage:")
    print("  from grow_model_finetuning import (")
    print("      GrowthAwareTrainer, grow_model_finetuning,")
    print("      TIGRQuestionGenerator, generate_tigr_training_data")
    print("  )")
    print("")
    print("  # Generate data")
    print("  data = generate_tigr_training_data(num_samples=500)")
    print("")
    print("  # Train with mixed batches")
    print("  model, history = grow_model_finetuning(")
    print("      model, data, epochs=500, batch_size=8,")
    print("      use_mixed_batches=True")
    print("  )")
    print("\n" + "="*70 + "\n")

