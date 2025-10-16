<?php
namespace App\Livewire\Institutions;

use Livewire\Component;
use App\Models\Institute;
use App\Models\Campus;
use LivewireUI\Modal\ModalComponent;

class InstituteCampusManager extends Component
{
    public $search = '';
    public $selectedInstituteId = '';
    public $institutes = [];
    public $selectedInstitute = null;
    public $allInstitutesWithCampuses = [];

    protected $listeners = [
        'refreshInstitutes' => 'loadData'
    ];

    public function mount()
    {
        $this->loadData();
    }

    public function loadData()
    {
        // Load all institutes for the dropdown
        $this->institutes = Institute::active()
            ->when($this->search, function($query) {
                $query->where('name', 'like', '%' . $this->search . '%');
            })
            ->orderBy('name')
            ->get();

        // Load all institutes with campuses for display
        $this->allInstitutesWithCampuses = Institute::with(['campuses' => function($query) {
            $query->active();
        }])->active()->get();

        // Auto-select first institute if none selected and institutes exist
        if (empty($this->selectedInstituteId) && $this->institutes->count() > 0) {
            $this->selectedInstituteId = $this->institutes->first()->id;
            $this->selectInstitute();
        } else if ($this->selectedInstituteId) {
            $this->selectInstitute();
        }
    }

    public function updatedSearch()
    {
        $this->loadData();
    }

    public function updatedSelectedInstituteId()
    {
        $this->selectInstitute();
    }

    public function selectInstitute()
    {
        if ($this->selectedInstituteId) {
            $this->selectedInstitute = Institute::with(['campuses' => function($query) {
                $query->active();
            }])->find($this->selectedInstituteId);
        } else {
            $this->selectedInstitute = null;
        }
    }

    public function addInstitute()
    {
        $this->dispatch('openModal', 'institutions.institute-modal');
    }

    public function editInstitute($instituteId)
    {
        $this->dispatch('openModal', 'institutions.institute-modal', ['instituteId' => $instituteId]);
    }

    public function addCampus($instituteId = null)
    {
        $instituteId = $instituteId ?: $this->selectedInstituteId;
        $this->dispatch('openModal', 'institutions.campus-modal', ['instituteId' => $instituteId]);
    }

    public function editCampus($campusId)
    {
        $this->dispatch('openModal', 'institutions.campus-modal', ['campusId' => $campusId]);
    }

    public function deleteInstitute($instituteId)
    {
        try {
            $institute = Institute::findOrFail($instituteId);
            
            // Check if institute has campuses
            if ($institute->campuses()->count() > 0) {
                $this->dispatch('showToast', [
                    'type' => 'error',
                    'message' => 'Cannot delete institute. It has associated campuses.',
                    'position' => 'top-end',
                    'duration' => 5000
                ]);
                return;
            }

            $institute->delete();
            $this->loadData();
            
            $this->dispatch('showToast', [
                'type' => 'success',
                'message' => 'Institute deleted successfully!',
                'position' => 'top-end',
                'duration' => 5000
            ]);
        } catch (\Exception $e) {
            $this->dispatch('showToast', [
                'type' => 'error',
                'message' => 'Failed to delete institute: ' . $e->getMessage(),
                'position' => 'top-end',
                'duration' => 5000
            ]);
        }
    }

    public function deleteCampus($campusId)
    {
        try {
            $campus = Campus::findOrFail($campusId);
            $campus->delete();
            $this->loadData();
            
            $this->dispatch('showToast', [
                'type' => 'success',
                'message' => 'Campus deleted successfully!',
                'position' => 'top-end',
                'duration' => 5000
            ]);
        } catch (\Exception $e) {
            $this->dispatch('showToast', [
                'type' => 'error',
                'message' => 'Failed to delete campus: ' . $e->getMessage(),
                'position' => 'top-end',
                'duration' => 5000
            ]);
        }
    }

    public function render()
    {
        return view('livewire.institutions.institute-campus-manager');
    }
}