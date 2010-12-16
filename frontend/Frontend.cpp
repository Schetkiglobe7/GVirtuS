/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   Frontend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:57:11 2009
 *
 * @brief
 *
 *
 */

#include "Frontend.h"

#include <iostream>
#include <string>

#include "ConfigFile.h"

using namespace std;

static Frontend msFrontend;
static bool initialized = false;

/**
 *
 */
void Frontend::Init() {
    const char *config_file;
#ifdef _CONFIG_FILE
    if((config_file = getenv("CONFIG_FILE")) == NULL)
        config_file = _CONFIG_FILE;
#else
	config_file = "gvirtus.properties";
#endif
    ConfigFile *cf = new ConfigFile(config_file);
    string communicator;
#ifndef _WIN32
    char *tmp;
	if((tmp = getenv("COMMUNICATOR")) != NULL)
        communicator = string(tmp);
    else
#endif
        communicator = cf->Get("communicator");
    mpCommunicator = Communicator::Get(communicator);
    mpCommunicator->Connect();
    mpInputBuffer = new Buffer();
    mpOutputBuffer = new Buffer();
    mpLaunchBuffer = new Buffer();
    mExitCode = cudaErrorUnknown;
    mpVar = new vector<CudaUtil::CudaVar *>();
    mAddingVar = false;
}

Frontend::~Frontend() {
    mpCommunicator->Close();
    delete mpCommunicator;
}

Frontend * Frontend::GetFrontend(bool register_var) {
    if (!initialized) {
        try {
            msFrontend.Init();
            initialized = true;
        } catch (const char *e) {
            cerr << "Error: cannot create Frontend ('" << e << "')" << endl;
        }
    }
    msFrontend.Prepare();
    if (msFrontend.mAddingVar && !register_var) {
        for (vector<CudaUtil::CudaVar *>::iterator it = msFrontend.mpVar->begin();
                it != msFrontend.mpVar->end(); it++)
            msFrontend.AddHostPointerForArguments(*it);
        msFrontend.Execute("cudaRegisterVar");
        msFrontend.mAddingVar = false;
        msFrontend.Prepare();
    }
    return &msFrontend;
}

void Frontend::Execute(const char* routine, const Buffer* input_buffer) {
    if (input_buffer == NULL)
        input_buffer = mpInputBuffer;

    /* sending job */
    mpCommunicator->Write(routine, strlen(routine) + 1);
    input_buffer->Dump(mpCommunicator);
    mpCommunicator->Sync();

    /* receiving output */
    //std::istream &in = mpCommunicator->GetInputStream();

    mpOutputBuffer->Reset();

    mpCommunicator->Read((char *) & mExitCode, sizeof (cudaError_t));
    size_t out_buffer_size;
    mpCommunicator->Read((char *) & out_buffer_size, sizeof (size_t));
    if (out_buffer_size > 0)
        mpOutputBuffer->Read<char>(mpCommunicator, out_buffer_size);
}

void Frontend::Prepare() {
    mpInputBuffer->Reset();
}
