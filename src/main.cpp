/**
 *
 */

// stdlib
#include <iostream>

// vtk
#include <vtkJPEGWriter.h>
#include <vtkNew.h>
#include <vtkOpenGLRenderer.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOSPRayPass.h>
#include <vtkOSPRayRendererNode.h>
#include <vtkRenderer.h>
#include <vtkWindowToImageFilter.h>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const int Width = 512;
    const int Height = 512;

    using RenderWindow = vtkRenderWindow;
    vtkNew<RenderWindow> renderWindow;
    renderWindow->EraseOn();
    renderWindow->ShowWindowOff();
    renderWindow->UseOffScreenBuffersOn();
    renderWindow->SetSize(Width, Height);
    renderWindow->SetMultiSamples(0);

    using RenderPass = vtkOSPRayPass;
    vtkNew<RenderPass> renderPass;

    using Renderer = vtkOpenGLRenderer;
    vtkNew<Renderer> renderer;
    renderer->SetBackground(0.2, 0.2, 0.2);
    renderer->SetPass(renderPass);

    renderWindow->AddRenderer(renderer);
    renderer->SetRenderWindow(renderWindow);

    std::cerr << (*renderWindow) << std::endl;

    using WindowToImageFilter = vtkWindowToImageFilter;
    vtkNew<WindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->Update();

    using JPEGWriter = vtkJPEGWriter;
    vtkNew<JPEGWriter> jpegWriter;
    jpegWriter->SetFileName("tmp/out.jpg");
    jpegWriter->SetInputConnection(windowToImageFilter->GetOutputPort());
    jpegWriter->Write();

    return 0;
}
