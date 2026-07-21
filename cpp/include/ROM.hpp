#pragma once

// Compatibility include for PETSc ROM classes.
// PETSc ROM classes live in namespace petsc_backend.
// New code should include Petsc_backend/ROM/... or ROM/Common/... directly.
#include "Petsc_backend/ROM/ROM_method/ROM_Base.hpp"
#include "Petsc_backend/ROM/ROM_method/FOM_direct.hpp"
#include "Petsc_backend/ROM/ROM_method/FOM_iterative.hpp"
#include "Petsc_backend/ROM/ROM_method/MyStrategy.hpp"
