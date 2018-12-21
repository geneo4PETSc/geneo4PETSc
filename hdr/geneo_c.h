#ifndef __GENEO_C_H
#define __GENEO_C_H
#include <petsc.h>
/*
 * createGenEOPC: create GenEO PC.
 * This function must be used as a callback passed to PCRegister.
 */

PETSC_EXTERN PetscErrorCode createGenEOPC(PC pcPC);
PETSC_EXTERN PetscErrorCode PCGenEOSetup(PC pc, IS, IS*);
#endif
