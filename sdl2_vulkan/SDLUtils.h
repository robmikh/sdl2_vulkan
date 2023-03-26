#pragma once

struct SDLDeleter
{
    void operator()(SDL_Window *p) const { SDL_DestroyWindow(p); }
};

typedef std::unique_ptr<SDL_Window, SDLDeleter> SDLWindow;
