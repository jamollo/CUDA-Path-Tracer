#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
// Always allocate. Restore uploads the already-restored host accumulation.
void pathtraceInit(Scene *scene, bool restoreAccumulation = false);
void pathtraceFree();
void pathtraceDisplay(uchar4 *pbo, int completedSamples);
void pathtrace(uchar4 *pbo, int frame, int iteration);
