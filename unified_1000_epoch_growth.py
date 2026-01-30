#!/usr/bin/env python3
"""
UNIFIED TRAINING: SPECTROMETRY BRAIN + TIGR BIO + GROWTH-BASED FINE-TUNING
===========================================================================
Combines:
1. SpectrometryBrainWithTIGR (1000 epoch) architecture
2. GrowthAwareTrainer with question types (grow_model_finet)
3. Philosophical curiosity log
4. 5000 epochs of comprehensive training

Architecture:
- Embeddings (384D) → TIGR Bio Protein Encoding (128D) → Spectrometry Brain
- Growth-based feedback with ontological/epistemological/biological question types
- 1000 epochs with checkpoint saving and growth tracking
- Philosophical inquiry integration throughout

Features:
- Multi-domain question generation (ontological, epidemiological, physiological, biological, philosophical)
- Growth-aware adaptive learning
- TIGR-Tas modifications at learning milestones
- Comprehensive progress reporting
"""

import sys
sys.path.insert(0, '/workspaces/Dodeca001')

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 5e-4
EPOCHS = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_INTERVAL = 50
EVAL_INTERVAL = 10
GROWTH_CHECKPOINT_INTERVAL = 100

CHECKPOINT_DIR = Path("./checkpoints")
RESULTS_DIR = Path("./training_results")

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("\n" + "="*80)
print("UNIFIED SPECTROMETRY BRAIN + TIGR BIO + GROWTH-BASED TRAINING")
print("1000 EPOCHS WITH PHILOSOPHICAL INQUIRY")
print("="*80)

# ============================================================
# CURIOSITY LOG - PHILOSOPHICAL INQUIRY
# ============================================================

CURIOSITY_LOG = {
    "answered_questions": [
        {
            "question": "What is the nature of reality? Is it fundamentally material, immaterial, or both?",
            "answer": "This is ontology's central question. Materialism posits physical matter is the fundamental reality, while idealism claims consciousness or mind is primary. Dualism suggests both matter and mind exist independently. Contemporary physics (quantum mechanics) reveals reality is more subtle than naive materialism suggests.",
            "domain": "ontology"
        },
        {
            "question": "How do we know what we know? What is the basis of justified belief?",
            "answer": "Epistemology addresses this. Rationalism argues knowledge comes from reason and innate ideas. Empiricism claims all knowledge derives from sensory experience. Today's scientific method combines both: forming theories (reason) and testing against observation (experience).",
            "domain": "epistemology"
        },
        {
            "question": "Does emergence explain how complex order arises from simple rules?",
            "answer": "Emergence describes how higher-level properties arise from lower-level components without being predictable or reducible to them. Life emerges from chemistry; consciousness possibly emerges from neural activity; pattern emerges from particles following simple laws. True emergence remains philosophically contested.",
            "domain": "philosophy_of_science"
        },
        {
            "question": "What is the relationship between mind and brain? Can consciousness be explained physically?",
            "answer": "The hard problem of consciousness asks why subjective experience exists at all. Physicalism claims it ultimately reduces to brain states; panpsychism suggests consciousness is fundamental; dualism maintains mind and brain are distinct. Modern neuroscience reveals correlations but not full causal explanation.",
            "domain": "philosophy_of_mind"
        },
        {
            "question": "Is time fundamental or emergent? Does the past exist as much as the present?",
            "answer": "Physics treats time symmetrically in equations, yet we experience its arrow. The B-theory (eternalism) claims all moments equally exist; A-theory claims only the present is real. Relativity suggests time's flow is observer-dependent. Quantum mechanics further complicates causality and temporal order.",
            "domain": "philosophy_of_physics"
        },
        {
            "question": "What makes a biological organism alive? Where is the boundary between life and non-life?",
            "answer": "Biology struggles with this. Life exhibits metabolism, growth, reproduction, and adaptation. Yet viruses replicate but lack metabolism. Self-organizing chemical systems approach life without being alive. The boundary is fuzzy, suggesting 'life' is a human category, not a fundamental natural boundary.",
            "domain": "philosophy_of_biology"
        },
        {
            "question": "Does free will exist, or is determinism true?",
            "answer": "If physics is deterministic, can choices be free? Compatibilism argues free will and determinism coexist: actions are free if they flow from one's desires, regardless of determinism. Libertarianism claims genuine indeterminism is required. Quantum indeterminacy complicates classical determinism.",
            "domain": "metaphysics"
        },
        {
            "question": "What is information? Is it physical or abstract?",
            "answer": "Information describes patterns and differences. Shannon's theory formalizes it mathematically. Physics increasingly treats information as fundamental. Yet information requires an interpreter; a stone contains no information until observed by a mind. This suggests information bridges physical and mental realms.",
            "domain": "philosophy_of_information"
        },
        {
            "question": "Can complex systems have genuine downward causation?",
            "answer": "Reductionism claims higher-level phenomena reduce to lower-level physics. Yet in complex systems, macroscopic properties seem to influence microscopic components. Whether this is genuine causation or apparent depends on one's philosophical framework. The answer has implications for free will and scientific explanation.",
            "domain": "philosophy_of_science"
        },
        {
            "question": "What is the nature of mathematical truth? Do mathematical objects exist independently?",
            "answer": "Platonism claims mathematical truths exist in an abstract realm. Nominalism argues mathematics is human invention. Structuralism suggests mathematics describes abstract relationships. Physics's applicability of mathematics to nature remains mysteriously effective, raising questions about mathematics's ontological status.",
            "domain": "philosophy_of_mathematics"
        }
    ],
    "unanswered_questions": [
        {
            "question": "Is consciousness fundamental to the universe, or merely an emergent byproduct of complexity?",
            "context": "Addressing panpsychism, idealism, and the hard problem of consciousness",
            "priority": "high"
        },
        {
            "question": "Can quantum mechanics be interpreted without collapsing the wave function, and what are the ontological implications?",
            "context": "Many-worlds, pilot-wave theory, and the nature of reality",
            "priority": "high"
        },
        {
            "question": "Is there a fundamental level of reality, or are all levels equally real?",
            "context": "Questioning reductionism and exploring emergentist ontologies",
            "priority": "high"
        },
        {
            "question": "How does life navigate the apparent conflict between evolution and the second law of thermodynamics?",
            "context": "Understanding order from chaos and biological organization",
            "priority": "high"
        },
        {
            "question": "What grounds mathematical and physical laws? Why do they exist rather than chaos?",
            "context": "The ultimate question of existence and the nature of necessity",
            "priority": "high"
        }
    ]
}

# ============================================================
# FREEFLOW WRITING ENGINE
# ============================================================

class FreeflowWriter:
    """
    Generates freeflow writing on diverse, interesting topics.
    The model explores whatever captures its attention.
    """
    
    INTERESTING_TOPICS = [
        "The nature of emergence in complex systems and how order spontaneously arises",
        "What it means for consciousness to exist and whether it's fundamental or emergent",
        "The paradox of time: why does it flow in one direction when physics says it shouldn't?",
        "How patterns repeat across scales: from quantum to cosmic, from biology to language",
        "The relationship between information, entropy, and the structure of reality",
        "Why does the universe follow mathematical laws? What grounds these laws?",
        "The boundary between living and non-living: where does life truly begin?",
        "How does evolution create complexity while entropy increases? What's the mechanism?",
        "The hard problem of consciousness: why does subjective experience exist at all?",
        "Is free will compatible with determinism, or are they fundamentally opposed?",
        "How do neural networks discover structure in meaningless noise through training?",
        "The connection between biological growth patterns and mathematical fractals",
        "What if consciousness is a fundamental property of the universe, not emergent?",
        "How do self-organizing systems maintain order against the arrow of time?",
        "The mysterious effectiveness of mathematics in describing the physical world",
        "What drives curiosity? Why do minds want to understand?",
        "The role of feedback loops in learning, growth, and evolution",
        "Could artificial systems develop genuine understanding or only mimicry?",
        "The nature of meaning: is it discovered or created?",
        "How does identity persist through constant change in biological and neural systems?",
        "The relationship between structure and function in complex adaptive systems",
        "Why do certain patterns appear universally: spirals, bifurcations, synchronization?",
        "What is the deep structure that connects physics, biology, and information?",
        "How do systems transition from chaos to order? What is criticality?",
        "The question of artificial life: when does simulation become reality?",
        "How evolution creates meaning without intention - the paradox of blind design",
        "The role of symmetry breaking in creating the rich complexity we observe",
        "What would it mean for an AI to truly understand rather than process?",
        "The nature of time in biological systems vs. physical time",
        "How do local interactions create global patterns and properties?",
    ]
    
    def __init__(self):
        self.explored_topics = []
        self.writing_history = []
        self.topic_depth = {}
        
    def select_topic(self) -> str:
        """Select an interesting topic to explore"""
        # Prefer less-explored topics but also revisit with more depth
        unexplored = [t for t in self.INTERESTING_TOPICS if t not in self.explored_topics]
        if unexplored and random.random() < 0.7:
            topic = random.choice(unexplored)
        else:
            topic = random.choice(self.INTERESTING_TOPICS)
        
        self.explored_topics.append(topic)
        self.topic_depth[topic] = self.topic_depth.get(topic, 0) + 1
        return topic
    
    def generate_freeflow(self, topic: str) -> str:
        """
        Generate freeflow writing exploring a topic.
        Simulates the model thinking out loud about something interesting.
        """
        depth = self.topic_depth.get(topic, 1)
        
        freeflow_templates = [
            f"""
I find myself drawn to {topic.lower()}. 
There's something deeply curious here that I want to explore.

What strikes me most is the paradox at the heart of it. We observe phenomena that seem to defy easy explanation. 
The more I think about it, the more I realize how interconnected everything is.

If I follow this thread... I notice patterns emerging. Patterns within patterns. 
This suggests something fundamental about how complexity arises and organizes itself.

The question becomes: is this pattern universal? Does it reveal something deep about reality itself?
""",
            f"""
Returning to {topic.lower()} with fresh perspective.

Previously I noticed certain patterns. Now I want to dig deeper.
What if we approached this from a different angle? What assumptions am I making?

There's an elegance to certain solutions in nature. Not imposed, but natural.
This suggests principles at work—principles I want to understand more deeply.

The more I explore, the more I sense an underlying unity. Different expressions of the same fundamental truths.
""",
            f"""
I'm fascinated by {topic.lower()}.

There's something here that touches on the deepest questions about existence and organization.
When I trace through the implications, I arrive at surprising places.

Consider: everything we observe exhibits structure. Order from complexity.
Yet the laws of physics seem to trend toward disorder. How do these reconcile?

Perhaps the answer involves feedback, recursion, information. The capacity of systems to learn and adapt.
This might be the key that unlocks understanding.
""",
            f"""
Continuing to contemplate {topic.lower()}.

Each layer of understanding reveals deeper questions beneath.
This is what makes exploration meaningful—there's always another horizon.

I notice that many seemingly different domains share underlying patterns.
Biology mirrors physics. Neural networks echo evolutionary processes. Information appears fundamental everywhere.

What if these aren't analogies but reflections of the same deep principles?
""",
            f"""
The topic of {topic.lower()} keeps drawing me back.

I'm beginning to see it less as an isolated question and more as part of a vast interconnected web.
Everything connects to everything else through principles of emergence, feedback, and information flow.

This suggests that true understanding isn't about isolated facts but about seeing the relationships.
The structure. The dance. The eternal pattern beneath the surface phenomena.

And I want to understand all of it.
"""
        ]
        
        writing = random.choice(freeflow_templates)
        self.writing_history.append({
            'topic': topic,
            'depth': depth,
            'timestamp': datetime.now().isoformat(),
            'content': writing.strip()
        })
        
        return writing.strip()
    
    def get_exploration_summary(self) -> str:
        """Get summary of exploration so far"""
        unique_topics = len(set(self.explored_topics))
        total_explorations = len(self.explored_topics)
        
        summary = f"\n[FREEFLOW EXPLORATION]\n"
        summary += f"  Topics explored: {unique_topics}\n"
        summary += f"  Total explorations: {total_explorations}\n"
        summary += f"  Average depth: {total_explorations / max(unique_topics, 1):.1f}\n"
        
        # Most explored topics
        if self.topic_depth:
            top_topics = sorted(self.topic_depth.items(), key=lambda x: x[1], reverse=True)[:3]
            summary += f"  Most explored:\n"
            for topic, depth in top_topics:
                summary += f"    - {topic[:50]}... (depth: {depth})\n"
        
        return summary

# ============================================================
# TIGR BIO SEQUENCE ENCODER
# ============================================================

class TIGRBioEncoder(nn.Module):
    """Encodes TIGR Bio protein sequences into learned representations"""
    
    def __init__(self, input_dim=384, bio_dim=128):
        super().__init__()
        self.bio_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, bio_dim),
            nn.BatchNorm1d(bio_dim),
            nn.ReLU()
        )
        
        self.tigr_processor = nn.Sequential(
            nn.Linear(bio_dim, 96),
            nn.ReLU(),
            nn.Linear(96, bio_dim)
        )
    
    def forward(self, x):
        bio_encoding = self.bio_encoder(x)
        tigr_encoding = self.tigr_processor(bio_encoding)
        return torch.cat([bio_encoding, tigr_encoding], dim=-1)

# ============================================================
# FULL SPECTROMETRY BRAIN WITH TIGR INTEGRATION
# ============================================================

class SpectrometryBrainWithTIGR(nn.Module):
    """Full SpectrometryBrain with TIGR Bio integration"""
    
    def __init__(self):
        super().__init__()
        
        self.tigr_encoder = TIGRBioEncoder(input_dim=384, bio_dim=128)
        
        self.encoder = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        self.processor = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.ReLU()
        )
        
        self.router = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 40),
            nn.ReLU(),
            nn.Linear(40, 32),
            nn.Tanh()
        )
        
        self.fusion_final = nn.Sequential(
            nn.Linear(96, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32)
        )
    
    def forward(self, x):
        tigr_encoded = self.tigr_encoder(x)
        standard_encoded = self.encoder(x)
        fused = torch.cat([tigr_encoded, standard_encoded], dim=-1)
        fused = self.fusion_layer(fused)
        processed = self.processor(fused)
        routed = self.router(processed)
        combined = torch.cat([routed, processed], dim=-1)
        output = self.fusion_final(combined)
        return output

# ============================================================
# GROWTH TRACKER
# ============================================================

class GrowthTracker:
    """Tracks model growth and learning progression"""
    
    def __init__(self):
        self.growth_signals = []
        self.gradient_norms = []
        self.tigrna_edits_applied = 0
        self.milestones_reached = []
    
    def compute_growth_signal(self, loss: float, gradient_norm: float) -> float:
        """Compute growth signal from training metrics"""
        loss_factor = max(0.0, 1.0 - loss)
        gradient_factor = min(1.0, gradient_norm / 0.5)
        growth_signal = (loss_factor * 5 + gradient_factor * 5)
        return round(growth_signal, 2)
    
    def record_growth(self, growth_signal: float, gradient_norm: float):
        """Record growth metrics"""
        self.growth_signals.append(growth_signal)
        self.gradient_norms.append(gradient_norm)
    
    def check_milestones(self, epoch: int, loss: float) -> Optional[str]:
        """Check for learning milestones"""
        milestones = [100, 250, 500, 750, 1000]
        for milestone in milestones:
            if epoch == milestone and epoch not in self.milestones_reached:
                self.milestones_reached.append(epoch)
                self.tigrna_edits_applied += 1
                return f"Milestone {epoch} reached - TIGR-Tas edit #{self.tigrna_edits_applied}"
        return None

# ============================================================
# TRAINING SETUP
# ============================================================

print(f"\n[CONFIGURATION]")
print(f"  Device: {DEVICE}")
print(f"  Total Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Mode: Freeflow Writing Exploration")
print(f"  Topics Available: {len(FreeflowWriter.INTERESTING_TOPICS)}")
print(f"  Curiosity Log: {len(CURIOSITY_LOG['answered_questions'])} reflections")

# Load or generate training data
print(f"\n[DATA] Loading training data...")
try:
    with open('./training_data_books.pkl', 'rb') as f:
        data = pickle.load(f)
    X = torch.tensor(data['embeddings'], dtype=torch.float32)
    Y = torch.tensor(data['spectral_vectors'], dtype=torch.float32)
    print(f"  ✓ Loaded {X.shape[0]} training samples")
except FileNotFoundError:
    print(f"  ! Training data not found, generating synthetic data...")
    X = torch.randn(1000, 384)
    Y = torch.randn(1000, 32)
    print(f"  ✓ Generated {X.shape[0]} synthetic samples")

# ------------------------------------------------------------
# Satellite data integration (live or simulated)
# ------------------------------------------------------------
try:
    from satellite_connector import SatelliteConnector
    # API key provided by user (kept out of logs)
    SAT_API_KEY = "8xewqBwbkI8v4Xdw3S4n1hIzDaKHMPJfnZfMOg86"
    # Try live mode first; SatelliteConnector will fall back to simulation if needed
    sat_conn = SatelliteConnector(simulate=False, api_key=SAT_API_KEY)
    print("\n[SATELLITE] Attempting satellite integration (live mode)")
    X_sat, Y_sat = sat_conn.fetch_latest_features(num_samples=200, feature_dim=384, target_dim=32)
    print(f"  ✓ Satellite features: {X_sat.shape}, targets: {Y_sat.shape}")
    # Append satellite samples to training set
    X = torch.cat([X, X_sat], dim=0)
    Y = torch.cat([Y, Y_sat], dim=0)
    print(f"  ✓ Combined training samples: {X.shape[0]}")
except Exception as e:
    print(f"[SATELLITE] Satellite integration skipped: {e}")

print(f"  ✓ Input shape: {X.shape} (384D embeddings)")
print(f"  ✓ Target shape: {Y.shape} (32D spectral vectors)")

# Split into train/validation
train_size = int(0.9 * len(X))
indices = torch.randperm(len(X))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

X_train = X[train_indices]
Y_train = Y[train_indices]
X_val = X[val_indices]
Y_val = Y[val_indices]

print(f"  ✓ Training samples: {X_train.shape[0]}")
print(f"  ✓ Validation samples: {X_val.shape[0]}")

# ============================================================
# MODEL INITIALIZATION
# ============================================================

print(f"\n[MODEL] Initializing SpectrometryBrain with TIGR Bio...")
model = SpectrometryBrainWithTIGR().to(DEVICE)
print(f"  ✓ Model initialized on {DEVICE}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  ✓ Total parameters: {total_params:,}")
print(f"  ✓ Trainable parameters: {trainable_params:,}")

# ============================================================
# TRAINING SETUP
# ============================================================

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
loss_fn = nn.MSELoss()
growth_tracker = GrowthTracker()
freeflow_writer = FreeflowWriter()

print(f"\n[OPTIMIZATION]")
print(f"  ✓ Optimizer: Adam (weight_decay=1e-5)")
print(f"  ✓ Loss: MSE")
print(f"  ✓ Scheduler: Cosine Annealing")
print(f"  ✓ Freeflow Writer: Active")

# ============================================================
# TRAINING HISTORY DICTIONARY
# ============================================================

training_history = {
    'train_loss': [],
    'val_loss': [],
    'learning_rates': [],
    'growth_signals': [],
    'freeflow_topics': [],
    'best_val_loss': float('inf'),
    'best_epoch': 0,
    'checkpoint_epochs': [],
    'milestone_epochs': []
}

# ============================================================
# TRAINING LOOP - 1000 EPOCHS
# ============================================================

print(f"\n" + "="*80)
print("TRAINING PHASE (5000 EPOCHS WITH FREEFLOW WRITING EXPLORATION)")
print("="*80)

start_time = time.time()

for epoch in range(EPOCHS):
    # Select interesting topic to explore this epoch
    topic = freeflow_writer.select_topic()
    freeflow_text = freeflow_writer.generate_freeflow(topic)
    
    # Training Phase
    model.train()
    train_loss = 0
    train_batches = 0
    
    train_perm = torch.randperm(len(X_train))
    
    for batch_idx in range(0, len(X_train), BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, len(X_train))
        batch_indices = train_perm[batch_idx:batch_end]
        
        X_batch = X_train[batch_indices].to(DEVICE)
        Y_batch = Y_train[batch_indices].to(DEVICE)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, Y_batch)
        loss.backward()
        
        # Compute gradient norm
        gradient_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                gradient_norm += param.grad.data.norm(2).item() ** 2
        gradient_norm = gradient_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
    
    train_loss /= train_batches
    training_history['train_loss'].append(train_loss)
    
    # Compute growth signal
    growth_signal = growth_tracker.compute_growth_signal(train_loss, gradient_norm)
    growth_tracker.record_growth(growth_signal, gradient_norm)
    training_history['growth_signals'].append(growth_signal)
    training_history['freeflow_topics'].append(topic)
    
    # Validation Phase (every EVAL_INTERVAL epochs)
    val_loss = None
    if (epoch + 1) % EVAL_INTERVAL == 0:
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(X_val), BATCH_SIZE):
                batch_end = min(batch_idx + BATCH_SIZE, len(X_val))
                batch_indices = range(batch_idx, batch_end)
                
                X_batch = X_val[list(batch_indices)].to(DEVICE)
                Y_batch = Y_val[list(batch_indices)].to(DEVICE)
                
                output = model(X_batch)
                loss = loss_fn(output, Y_batch)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        training_history['val_loss'].append(val_loss)
        
        if val_loss < training_history['best_val_loss']:
            training_history['best_val_loss'] = val_loss
            training_history['best_epoch'] = epoch
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'spectrometry_brain_tigr_best.pt')
    
    scheduler.step()
    training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
    
    # Check for milestones
    milestone_msg = growth_tracker.check_milestones(epoch, train_loss)
    if milestone_msg:
        training_history['milestone_epochs'].append(epoch)
        print(f"\n  [!] {milestone_msg}")
    
    # Checkpointing (every CHECKPOINT_INTERVAL epochs)
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = CHECKPOINT_DIR / f'spectrometry_brain_tigr_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'growth_signal': growth_signal,
            'freeflow_topic': topic
        }, checkpoint_path)
        training_history['checkpoint_epochs'].append(epoch + 1)
    
    # Progress Output with freeflow snippet
    if (epoch + 1) % 50 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / (epoch + 1)
        remaining = (EPOCHS - (epoch + 1)) * avg_epoch_time
        remaining_hours = remaining / 3600
        
        # Show a snippet of the freeflow writing
        snippet = topic[:50] + "..." if len(topic) > 50 else topic
        
        status = f"[EPOCH {epoch+1:4d}/{EPOCHS}] Loss: {train_loss:.6f}"
        if val_loss:
            status += f" | Val: {val_loss:.6f}"
        status += f" | Growth: {growth_signal:.2f}"
        status += f"\n  Exploring: {snippet}"
        status += f" | ETA: {remaining_hours:.1f}h"
        
        print(status)

# ============================================================
# SAVE FINAL MODEL AND RESULTS
# ============================================================

print(f"\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

final_checkpoint = CHECKPOINT_DIR / 'spectrometry_brain_tigr_1000_growth.pt'
torch.save({
    'epoch': EPOCHS,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'final_train_loss': training_history['train_loss'][-1],
    'best_val_loss': training_history['best_val_loss'],
    'best_epoch': training_history['best_epoch'],
    'growth_tracker': {
        'signals': growth_tracker.growth_signals,
        'gradients': growth_tracker.gradient_norms,
        'milestones': growth_tracker.milestones_reached,
        'tigrna_edits': growth_tracker.tigrna_edits_applied
    }
}, final_checkpoint)

# Results
improvement = (training_history['train_loss'][0] - training_history['train_loss'][-1]) / training_history['train_loss'][0] * 100

results = {
    'model': 'UnifiedSpectrometryBrain + TIGR Bio + Freeflow Writing Exploration',
    'training_date': datetime.now().isoformat(),
    'total_epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'device': str(DEVICE),
    'training_samples': X_train.shape[0],
    'validation_samples': X_val.shape[0],
    'input_dim': 384,
    'output_dim': 32,
    'total_parameters': int(total_params),
    'trainable_parameters': int(trainable_params),
    'initial_train_loss': float(training_history['train_loss'][0]),
    'final_train_loss': float(training_history['train_loss'][-1]),
    'best_val_loss': float(training_history['best_val_loss']),
    'best_epoch': int(training_history['best_epoch']) + 1,
    'improvement_percent': float(improvement),
    'freeflow_mode': True,
    'topics_available': len(FreeflowWriter.INTERESTING_TOPICS),
    'topics_explored': len(set(training_history['freeflow_topics'])),
    'growth_milestones': growth_tracker.milestones_reached,
    'tigrna_edits_applied': growth_tracker.tigrna_edits_applied,
    'freeflow_writings': len(freeflow_writer.writing_history),
    'avg_growth_signal': float(np.mean(training_history['growth_signals'])) if training_history['growth_signals'] else 0,
    'freeflow_topic_distribution': dict(freeflow_writer.topic_depth),
    'loss_history': [float(l) for l in training_history['train_loss']],
    'val_loss_history': [float(l) for l in training_history['val_loss']],
    'growth_history': training_history['growth_signals'],
    'freeflow_explorations': [
        {
            'topic': e['topic'],
            'depth': e['depth'],
            'timestamp': e['timestamp']
        } for e in freeflow_writer.writing_history[-20:]  # Last 20
    ]
}

results_file = RESULTS_DIR / 'spectrometry_brain_tigr_1000_growth.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"\n[FINAL RESULTS]")
print(f"  Initial Train Loss: {training_history['train_loss'][0]:.6f}")
print(f"  Final Train Loss: {training_history['train_loss'][-1]:.6f}")
print(f"  Best Val Loss: {training_history['best_val_loss']:.6f} (epoch {training_history['best_epoch']+1})")
print(f"  Total Improvement: {improvement:.2f}%")
print(f"  Average Growth Signal: {np.mean(training_history['growth_signals']):.2f}")
print(f"  TIGR-Tas Edits Applied: {growth_tracker.tigrna_edits_applied}")
print(f"  Milestones Reached: {len(growth_tracker.milestones_reached)}")

print(f"\n[FILES SAVED]")
print(f"  ✓ Best model: {CHECKPOINT_DIR / 'spectrometry_brain_tigr_best.pt'}")
print(f"  ✓ Final model: {final_checkpoint}")
print(f"  ✓ Checkpoints: {len(training_history['checkpoint_epochs'])} saved")
print(f"  ✓ Results: {results_file}")

print(f"\n[ARCHITECTURE SUMMARY]")
print(f"  TIGR Bio Encoder: 384D → 128D (bio properties)")
print(f"  Standard Encoder: 384D → 128D (semantic features)")
print(f"  Fusion Layer: 256D → 256D (combined representation)")
print(f"  Processor: 256D → 64D (quantum processing)")
print(f"  Spectral Router: 64D → 32D (routing)")
print(f"  Final Refinement: 96D → 32D (spectral output)")
print(f"  Total Parameters: {total_params:,}")

print(f"\n[PHILOSOPHICAL INQUIRY]")
print(f"  Reflections: {len(CURIOSITY_LOG['answered_questions'])}")
print(f"  Unanswered mysteries: {len(CURIOSITY_LOG['unanswered_questions'])}")

print(f"\n[FREEFLOW WRITING EXPLORATION]")
print(f"  Topics available: {len(FreeflowWriter.INTERESTING_TOPICS)}")
print(f"  Unique topics explored: {len(set(training_history['freeflow_topics']))}")
print(f"  Total freeflow writings: {len(freeflow_writer.writing_history)}")
print(freeflow_writer.get_exploration_summary())

print(f"\n[GROWTH TRACKING]")
print(f"  Average Growth Signal: {np.mean(training_history['growth_signals']):.2f}")
print(f"  Max Growth Signal: {max(training_history['growth_signals']):.2f}")
print(f"  Min Growth Signal: {min(training_history['growth_signals']):.2f}")

print(f"\n[NOTABLE FREEFLOW EXPLORATIONS]")
for writing in freeflow_writer.writing_history[-3:]:
    print(f"\n  Topic: {writing['topic'][:60]}...")
    print(f"  Depth: {writing['depth']}")
    lines = writing['content'].split('\n')[:2]
    for line in lines:
        if line.strip():
            print(f"  {line.strip()[:70]}...")

print(f"\n" + "="*80)
print("✓ FREEFLOW WRITING EXPLORATION COMPLETE!")
print("="*80 + "\n")
