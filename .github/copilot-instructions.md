# Comprehensive Instructions for Entorhinal-Hippocampal Circuit Modeling

## Overview & Overall Goal

Develop a Python library to model the entorhinal-hippocampal circuit for spatial navigation and memory functions, including:

- Pattern separation: Decoding distinct cognitive maps from overlapping inputs
- Completion: Recovering full cognitive maps from partial inputs

This library will:

- Support training using Backpropagation (BP), DFA, and other biologically plausible learning rules
- Integrate key regions of the entorhinal-hippocampal circuit to simulate information flow
- Be distributed via PyPI (`pip install ehc-sn`)
- Provide network definition and parameters via TOML files loaded with Pydantic 2
- Focus on forward learning mechanisms and neural representations

## Project Structure

- `pyproject.toml` - Package configuration and build settings
- `src/ehc_sn/` - Main package source code
  - `VERSION` - File containing the current version number
  - `config/` - TOML files for library configuration (parameters, etc.)
  - `constants/` - Constants and types definitions (Enums, literals, types, etc.)
  - `core/` - Implementation of core library components (neurons, synapses, etc.)
  - `data/` - Data module for generating and managing Lightning data modules (cognitive maps, grid maps, etc.)
  - `figures/` - Module containing figure classes to visualize experiment results
  - `hooks/` - Custom PyTorch Lightning callbacks and hooks
  - `models/` - Implementation of library models (CANModel, Autoencoder, etc.)
  - `modules/` - Reusable PyTorch modules (layers, loss functions, etc.)
  - `trainers/` - Training routines and experiment management
  - `utils/` - Utility functions for the library
- `requirements-dev.txt` - Development dependencies
- `requirements.txt` - Core package dependencies
- `tests/` - Unit and integration tests

### Config Direct

- This directory should contain TOML files for library configuration
- It does not contain any code, only configuration files

### Constants Module

- This module should contain constants and types definitions
- It should include Enums, literals, and type definitions for the library
- It should be used to define constants for neuron types, synapse types, and other fixed parameters
- It should be used to define types for parameters and configurations

### Core Module

- This module should implement core library components such as neurons, synapses, and other fundamental building blocks
- It should include neuron models, synapse models, and other essential components for the entorhinal-hippocampal circuit
- It should provide a foundation for building more complex models and experiments
- It should be structured to allow easy extension and modification of core components

### Data Module

- Provide generatators and lightning data modules for obstacle and cognitive maps.
-

### Figures Module

- Provide functionality to visualize cognitive maps and model performance
- Implement visualization objects using Matplotlib and Seaborn
- Structure with:

1. Class for figure with plot method to generate plots
2. Consistent use of Pydantic for parameter management
3. Good handling of tensor conversion and visualization options
4. Comprehensive visualization capabilities for different map types

### Models Module

- This module should implement the library models, such as CANModel, Autoencoder, and other relevant models
- Each model should encapsulate the structure and behavior of the entorhinal-hippocampal circuit
- Models should be designed to support both training and inference modes
- Models should be modular and extensible to allow for future enhancements and variations

### Utils Module

- This module should contain utility functions for the library
- It should include helper functions for data processing, tensor manipulation, and other common tasks
- It should provide reusable functions that can be used across different modules
- It should be structured to allow easy addition of new utility functions as needed
- It should include functions for tensor conversion, data normalization, and other common operations
- It should be designed to minimize dependencies on other modules to ensure reusability
- It should provide functions for logging, debugging, and other common tasks that are not specific to any module
- It should include functions for handling configuration files, such as loading and validating TOML files with Pydantic
- It should provide functions for managing parameters and settings across the library

## Modeling Framework

- PyTorch with custom neuron models for the entorhinal-hippocampal circuit
- PyTorch Lightning for training, data loading and callbacks (but using manual optimization)
- Torchvision for data transformations and utilities
- Norse for spiking neuron models and event-based processing
- Optuna for hyperparameter optimization
- TorchRL for reinforcement learning components, if needed
- Keep code simple and avoid extending functionality beyond requirements
- When implementing modifications, minimize the amount of code removed or added

## Circuit Components and Properties

The models are structured as autoencoders, where:

- Medial Entorhinal Cortex (MEC) layers act as encoder for extracting features from sensory inputs
- Hippocampal regions act as a decoder for reconstructing cognitive maps

### Master EHC Wiring Summary Table

| Region / Layer                                    | Inputs                                                                  | Computations / Roles                                                                         | Outputs                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **External Inputs**                               |                                                                         |                                                                                              |                                                            |
| **Perirhinal Cortex (PER)**                       | Sensory cortices                                                        | Object identity, “what” info                                                                 | To **LEC II/III (plastic)**                                |
| **Postrhinal Cortex (POR)**                       | Visual/parietal                                                         | Scene-based spatial info, landmarks                                                          | To **MEC II/III, V/VI (plastic)**                          |
| **Retrosplenial Cortex (RSC)**                    | Visual/parietal, head direction system                                  | Head direction, egocentric–allocentric transformation                                        | To **MEC II/III, V/VI (plastic)**                          |
| **Medial Prefrontal Cortex (mPFC)**               | Association cortices                                                    | Task demands, goals, executive modulation                                                    | To **MEC V/VI, LEC V/VI (plastic)**                        |
| **Thalamus (anterodorsal, laterodorsal)**         | Subcortical head direction system                                       | Orientation, direction                                                                       | To **MEC I–III (plastic)**                                 |
| **Septum (MS/DBB)**                               | Subcortical modulatory                                                  | Theta rhythm, cholinergic/GABAergic modulation                                               | To **HPC + EC (plastic)**                                  |
| **Amygdala**                                      | Limbic                                                                  | Emotional/motivational salience                                                              | To **LEC II/III, V/VI (plastic)**                          |
| **Presubiculum**                                  | Head-direction system                                                   | Head direction signals                                                                       | To **MEC superficial (plastic)**                           |
| **Parasubiculum** Master EHC Wiring Summary Table | Entorhinal–hippocampal                                                  | Spatial boundaries, context                                                                  | To **MEC & LEC (plastic)**                                 |
| **LEC – Lateral Entorhinal Cortex**               |                                                                         |                                                                                              |                                                            |
| **LEC Layer II**                                  | From PER, amygdala, sensory cortices (plastic)                          | Encodes object, “what” features                                                              | To DG (plastic), CA3 (fixed), CA2 (fixed)                  |
| **LEC Layer III**                                 | From PER/multimodal cortices (plastic)                                  | Object–context associations, temporal sequence input                                         | To **CA1 (fixed)**, **Subiculum (fixed)**                  |
| **LEC Layer Va**                                  | From internal LEC processing (plastic)                                  | Integrates non-spatial context, sends to cortex                                              | To neocortex (plastic)                                     |
| **LEC Layer Vb**                                  | From Subiculum, CA1 (plastic)                                           | Integrates hippocampal output with object/context input                                      | To LEC II/III (fixed-weight relay)                         |
| **MEC – Medial Entorhinal Cortex (Encoder)**      |                                                                         |                                                                                              |                                                            |
| **MEC Layer II**                                  | From MEC Vb, POR/RSC, thalamus (plastic)                                | Position & trajectory coding; attractor dynamics                                             | To DG (plastic), **CA3 (fixed)**, **CA2 (fixed)**, MEC Vb  |
| **MEC Layer III**                                 | From MEC Vb, POR/RSC, thalamus (plastic)                                | Heading direction & speed coding; provides feedforward drive                                 | To **CA1 (fixed)**, **Subiculum (fixed)**                  |
| **MEC Layer Va**                                  | From internal MEC processing (plastic)                                  | Low-excitability integrator; hidden states; gateway to cortex                                | To neocortex (plastic)                                     |
| **MEC Layer Vb**                                  | From MEC II, Subiculum, CA1 (plastic)                                   | Integrates hippocampal reconstructions; hidden states; relay                                 | To MEC II/III (fixed-weight relay)                         |
| **Hippocampal Circuit (Decoder)**                 |                                                                         |                                                                                              |                                                            |
| **Dentate Gyrus (DG)**                            | From MEC II (plastic), LEC II (plastic)                                 | Sparse embeddings, pattern separation; neurogenesis expands coding                           | To CA3 (plastic)                                           |
| **CA3**                                           | From DG (plastic), **MEC II/LEC II (fixed)**                            | Attractor dynamics, recurrent collaterals; pattern completion; place cells                   | To CA2 (plastic), CA1 (plastic)                            |
| **CA2**                                           | From CA3 (plastic), **MEC II/LEC II (fixed)**                           | Specialized integration; SWR modulation; unique plasticity                                   | To CA1 (plastic)                                           |
| **CA1**                                           | From CA3 & CA2 (plastic), **MEC III/LEC III (fixed feedforward drive)** | Compares predictions (CA3) vs. sensory (EC); reconstructs cognitive map; contextual encoding | To MEC Vb (plastic), LEC Vb (plastic), Subiculum (plastic) |
| **Subiculum**                                     | From CA1 (plastic), **MEC III/LEC III (fixed)**                         | Major hippocampal output hub; border cells, head direction, grid-like coding                 | To MEC Vb, LEC Vb (plastic); neocortex                     |

## Models and Experiments

- First model should be a general sparse autoencoder to use as baseline
- Initial models replace the MEC by a standard sparse encoder trained a priori on cognitive maps
- Initial models ignore the Subiculum layer to work with one reconstruction at a time
- Full models will include the Subiculum layer and the MEC encoder layers

## Code Specifications

- Use type hints for function signatures
- Use Pydantic v2 for parameter validation and management
- Import in `__init__.py` from submodules to define the package/subpackage API
- Target Python 3.10 and newer (3.10, 3.11, 3.12)
- Prioritize vectorized operations over loops when applicable for performance

### Comments to split code sections

- Use comments with '-' to visually separate sections of the code
- Every function, class or method should have a section according to indentation
- Only 2 levels of separators are defined, one for the main section and one for subsections

```python
# -------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
```

## Documentation Guidelines

- Use docstrings for all public functions and classes
- Follow PEP 257 conventions for docstrings
- `__init__.py` files should contain package-level documentation
- `__init__.py` files should only contain import statements, package metadata, and documentation strings

## Evaluation Criteria and Metrics

Evaluate model performance using these metrics:

1. Reconstruction Accuracy:

- Mean squared error (MSE) between original and reconstructed maps
- Structural similarity index (SSIM) for spatial coherence

2. Pattern Separation:

- Discriminability index between representations of similar inputs
- Hamming distance between encodings of similar patterns

3. Pattern Completion:

- Accuracy of reconstruction from partial inputs (10%, 30%, 50% missing)
- Recovery time (iterations needed for stable completion)

4. Biological Plausibility:

- Sparsity of representations (percentage of active units)
- Activity distributions compared to neurobiological data

5. Computational Efficiency:

- Training time and convergence speed
- Memory usage during training and inference

Each experiment should report these metrics in a standardized format to allow for comparison between model variants.

## Library Domain Guidelines

- Use established terminology from hippocampal and spatial navigation literature
- Include relevant citations as comments where appropriate
- Optimize computationally intensive operations with appropriate algorithms
- Implement proper error handling for mathematical and scientific calculations

## Testing Guidelines

- Use pytest framework for all tests
- Design testing and configure pytest to use importlib import mode
- Write functional tests for all public functions and classes
- Mock complex neural simulations appropriately in tests
- Verify mathematical correctness with known solutions where possible
- Test edge cases relevant to neural modeling (boundary conditions, numerical stability)

## Packaging Guidelines

- Maintain version number in src/ehc_sn/VERSION file
- Keep core dependencies in requirements.txt
- Keep development dependencies in requirements-dev.txt
- Ensure compatibility with PyPI distribution standards
- Use dynamic configuration in pyproject.toml where appropriate

## General Guidelines

- Suggest code only relevant to the current task
- Avoid redundancies by referencing unchanged code with comments
- Update instructions and README as the library evolves
- Prevent unnecessary complexity by keeping the codebase clean and maintainable
- Prevent long functions by breaking them into smaller, reusable components
- Prevent deep conditions by breaking them into smaller, manageable functions
- Prevent deep loops by using vectorized operations and comprehensions
