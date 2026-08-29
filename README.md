# semantic-boundaries

Utilities for constructing boundaries around 2D semantic spaces.

## Installation

```bash
pip install semantic-boundaries
```
python -m venv .venv
source .venv/bin/activate
## Usage

```python
import numpy as np
from semantic_boundaries import boundary

P=np.column_stack([x,y])

x_with_boundary,x1,y1,xgrid,ygrid,h0=boundary(P)
```

P must be an (n,2) array containing x/y coordinates.

```python
boundary(
    P,
    grid=50,
    density=0.40,
    box_offset=0.1,
    tightness="auto"
)
```

Returns:
- x_with_boundary
- x1
- y1
- xgrid
- ygrid
- h0
